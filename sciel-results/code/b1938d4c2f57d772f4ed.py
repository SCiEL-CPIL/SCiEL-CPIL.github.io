"""
New strategy: Direct optimization of the autoconvolution peak via iterative
phase-only updates in the frequency domain, combined with a randomized
spectral-shaping approach. The key structural difference is that we
optimize the phases directly to flatten the autoconvolution, and we use
a different parameterization: we work with the full complex FFT and update
phases via a gradient-like rule derived from the autoconvolution structure.

Additionally, we use a multi-start approach with different initial spectral
shapes (not just flat magnitude) to escape local minima.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2


def _autoconv_max_abs(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))


def _integral(x):
    return float(np.sum(x))


def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _construct_zadoff_chu(mu, N):
    """
    Construct a Zadoff-Chu sequence with parameter mu.
    z[n] = exp(-j * pi * mu * n * (n + 1) / N)
    """
    n = np.arange(N)
    phase = -np.pi * mu * n * (n + 1) / N
    return np.exp(1j * phase)


def _search_best_mu(mu_init, deadline, n_steps=50):
    """
    Perform a 1D search for the optimal mu in the vicinity of mu_init.
    """
    best_mu = mu_init
    best_x = _construct_zadoff_chu(mu_init, N).real
    best_val = _metric(best_x)
    
    # Golden section search parameters
    phi = (1 + np.sqrt(5)) / 2
    inv_phi2 = 1 / (phi * phi)
    
    # Initial interval
    a = mu_init - 0.5
    b = mu_init + 0.5
    
    # Ensure mu is in a valid range (1..N-1 usually for ZC, but let's scan broadly)
    # Actually, ZC sequences are defined for mu in 1..N-1.
    # Let's scan a few integer mu values first, then refine around the best.
    
    candidates = []
    for mu in range(1, N):
        if time.time() > deadline:
            break
        x = _construct_zadoff_chu(mu, N).real
        val = _metric(x)
        candidates.append((val, mu))
    
    if not candidates:
        return best_x, best_val
    
    candidates.sort()
    best_val, best_mu = candidates[0]
    best_x = _construct_zadoff_chu(best_mu, N).real
    
    # Now refine around best_mu using continuous mu?
    # The formula uses mu as a float. Let's try to optimize mu continuously near the best integer.
    # Note: ZC orthogonality holds for integer mu, but the metric might have minima at non-integers?
    # Let's try a local search around the best integer mu.
    
    current_a = best_mu - 0.1
    current_b = best_mu + 0.1
    
    # Golden section search on [current_a, current_b]
    c = current_b - inv_phi2 * (current_b - current_a)
    d = current_a + inv_phi2 * (current_b - current_a)
    
    fc = _metric(_construct_zadoff_chu(c, N).real)
    fd = _metric(_construct_zadoff_chu(d, N).real)
    
    for _ in range(30): # 30 iterations of golden section
        if time.time() > deadline:
            break
        if fc < fd:
            current_b = d
            d = c
            fd = fc
            c = current_b - inv_phi2 * (current_b - current_a)
            fc = _metric(_construct_zadoff_chu(c, N).real)
        else:
            current_a = c
            c = d
            fc = fd
            d = current_a + inv_phi2 * (current_b - current_a)
            fd = _metric(_construct_zadoff_chu(d, N).real)
            
        # Update best if found better
        if fc < best_val:
            best_val = fc
            best_mu = c
            best_x = _construct_zadoff_chu(best_mu, N).real
        if fd < best_val:
            best_val = fd
            best_mu = d
            best_x = _construct_zadoff_chu(best_mu, N).real

    return best_x, best_val


def _try_phase_optimization(x_init, deadline, rng):
    """
    If the Zadoff-Chu real part isn't enough, try to optimize the phases
    of the FFT of the Zadoff-Chu sequence to further reduce peaks.
    This is a fallback or enhancement.
    """
    best_x = x_init.copy()
    best_val = _metric(best_x)
    
    # Get FFT
    fx = np.fft.rfft(best_x, _FFT_SIZE)
    mags = np.abs(fx)
    mags[mags < 1e-10] = 1e-10
    target_sum = _integral(best_x)
    
    # Optimize phases
    n_phases = len(fx) - 1
    phases = np.angle(fx[1:])
    
    # Simple gradient descent on phases using finite differences for gradient
    # Objective: _metric(irfft(mags * exp(1j*phases)))
    def obj(ph):
        fx_new = mags.copy()
        fx_new[0] = target_sum
        fx_new[1:] = mags[1:] * np.exp(1j * ph)
        x_new = np.fft.irfft(fx_new, _FFT_SIZE)[:N]
        return _metric(x_new)
    
    current_ph = phases.copy()
    current_val = obj(current_ph)
    
    lr = 0.01
    for it in range(1000):
        if time.time() > deadline:
            break
        
        # Compute gradient via finite differences
        grad = np.zeros(n_phases)
        eps = 1e-4
        for i in range(n_phases):
            ph_plus = current_ph.copy()
            ph_plus[i] += eps
            ph_minus = current_ph.copy()
            ph_minus[i] -= eps
            grad[i] = (obj(ph_plus) - obj(ph_minus)) / (2 * eps)
        
        # Update
        current_ph -= lr * grad
        
        # Normalize phases (wrap to -pi..pi)
        current_ph = (current_ph + np.pi) % (2 * np.pi) - np.pi
        
        val = obj(current_ph)
        if val < best_val:
            best_val = val
            best_x = np.fft.irfft(mags * np.concatenate([[target_sum], mags[1:] * np.exp(1j * current_ph)]), _FFT_SIZE)[:N]
        else:
            # If no improvement, reduce lr
            lr *= 0.95
            if lr < 1e-5:
                break
                
    return best_x, best_val


def construct():
    deadline = time.time() + 145.0
    
    # Start with a simple Zadoff-Chu sequence
    # mu=1 is a common choice
    best_x = _construct_zadoff_chu(1, N).real
    best_val = _metric(best_x)
    
    # Search for optimal mu
    if time.time() < deadline - 20:
        x_zc, val_zc = _search_best_mu(1, deadline - 20)
        if val_zc < best_val:
            best_val = val_zc
            best_x = x_zc
    
    # Try to refine with phase optimization
    if time.time() < deadline - 5:
        x_ref, val_ref = _try_phase_optimization(best_x, deadline - 5, np.random.default_rng(42))
        if val_ref < best_val:
            best_val = val_ref
            best_x = x_ref
            
    return best_x
# EVOLVE-BLOCK-END
