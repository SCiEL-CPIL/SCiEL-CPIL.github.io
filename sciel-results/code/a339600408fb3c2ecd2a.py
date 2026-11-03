"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Zero-pad size for linear autoconvolution
_M = 1
while _M < 2 * N - 1:
    _M *= 2

def _compute_linear_autoconv(h):
    """Compute linear autoconvolution of h with itself."""
    Hp = np.fft.rfft(h, _M)
    ac = np.fft.irfft(Hp * Hp, _M)
    return ac[:2 * N - 1]

def _score(h):
    """Compute max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _compute_linear_autoconv(h)
    return np.max(np.abs(ac)) / (s * s)

def _freq_to_time(x):
    """Convert frequency domain vector to time domain.
    x[0] = Re(H[0]), x[1] = Im(H[0]) [unused, set 0]
    For k=1..N-1: x[1+2k-1] = Re(H[k]), x[1+2k] = Im(H[k])
    H[N-k] = conj(H[k])
    """
    n_half = N // 2
    H = np.zeros(N, dtype=complex)
    H[0] = x[0] + 1j * x[1]
    for k in range(1, n_half + 1):
        H[k] = x[1 + 2*(k-1)] + 1j * x[1 + 2*(k-1) + 1]
        H[N - k] = np.conj(H[k])
    if N % 2 == 0:
        H[N//2] = x[1 + 2*(n_half-1)] + 1j * x[1 + 2*(n_half-1) + 1]
    return np.fft.ifft(H).real

def _time_to_freq(h):
    """Convert time domain to frequency domain vector x."""
    H = np.fft.fft(h)
    n_half = N // 2
    x = np.zeros(1 + 2 * n_half)
    x[0] = H[0].real
    x[1] = H[0].imag
    for k in range(1, n_half + 1):
        x[1 + 2*(k-1)] = H[k].real
        x[1 + 2*(k-1) + 1] = H[k].imag
    return x

def _smooth_max(x, beta=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

def _freq_objective(x, beta=100.0):
    """Objective in frequency domain with smooth max surrogate."""
    h = _freq_to_time(x)
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _compute_linear_autoconv(h)
    abs_ac = np.abs(ac)
    smooth_max_val = _smooth_max(abs_ac, beta)
    return smooth_max_val / (s * s)

def _freq_optimize():
    """Optimize in frequency domain using gradient-based methods."""
    t_start = time.time()
    deadline = t_start + 85.0
    
    n_half = N // 2
    n_vars = 1 + 2 * n_half
    
    best_score = np.inf
    best_x = None
    
    # Initializations
    initializations = []
    
    # 1. Flat: h = 1/N -> H[0]=1, H[k]=0
    x_flat = np.zeros(n_vars)
    x_flat[0] = 1.0
    initializations.append(x_flat)
    
    # 2. DC only with different scales
    for scale in [0.5, 1.0, 2.0, 5.0]:
        x_dc = np.zeros(n_vars)
        x_dc[0] = scale
        initializations.append(x_dc)
    
    # 3. Decaying spectrum
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
        x_init = np.zeros(n_vars)
        x_init[0] = 1.0
        for k in range(1, n_half + 1):
            x_init[1 + 2*(k-1)] = 0.05 / (1.0 + k**alpha)
        initializations.append(x_init)
    
    # 4. Random perturbations
    rng = np.random.default_rng(42)
    for _ in range(12):
        x_init = np.zeros(n_vars)
        x_init[0] = 1.0
        x_init[1:] = rng.normal(0, 0.03, size=2*n_half)
        initializations.append(x_init)
    
    # 5. Alternating sign in time domain
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    h_alt[1::2] = -1.0
    initializations.append(_time_to_freq(h_alt))
    
    # 6. Block patterns
    for block_size in [2, 4, 8, 16]:
        h_block = np.zeros(N)
        for i in range(0, N, block_size):
            sign = 1.0 if (i // block_size) % 2 == 0 else -1.0
            h_block[i:i+block_size] = sign
        initializations.append(_time_to_freq(h_block))
    
    for x_init in initializations:
        if time.time() > deadline - 15:
            break
        
        # Evaluate initial
        h_init = _freq_to_time(x_init)
        score_init = _score(h_init)
        if score_init < best_score:
            best_score = score_init
            best_x = x_init.copy()
        
        # Optimize
        try:
            result = minimize(
                _freq_objective,
                x_init,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            h_res = _freq_to_time(result.x)
            score_res = _score(h_res)
            
            if score_res < best_score:
                best_score = score_res
                best_x = result.x.copy()
        except Exception:
            continue
    
    if best_x is None:
        return np.ones(N) / N, _score(np.ones(N) / N)
    
    h = _freq_to_time(best_x)
    return h, best_score

def _local_refine(h, t_deadline):
    """Local refinement with sign flips and small perturbations."""
    h_refined = h.copy()
    best_local = _score(h_refined)
    
    # Sign flip refinement
    for _ in range(3):
        improved = False
        for idx in range(N):
            if time.time() > t_deadline - 5:
                return h_refined, best_local
            h_refined[idx] = -h_refined[idx]
            new_val = _score(h_refined)
            if new_val < best_local:
                best_local = new_val
                improved = True
            else:
                h_refined[idx] = -h_refined[idx]
        if not improved:
            break
    
    # Small random perturbations
    rng = np.random.default_rng(123)
    for _ in range(50):
        if time.time() > t_deadline - 3:
            break
        h_trial = h_refined.copy()
        idx = rng.integers(0, N)
        h_trial[idx] += rng.normal(0, 0.1 * max(1.0, np.max(np.abs(h_refined))))
        new_val = _score(h_trial)
        if new_val < best_local:
            best_local = new_val
            h_refined = h_trial
    
    return h_refined, best_local

def construct():
    t_start = time.time()
    deadline = t_start + 115.0
    
    best_score = np.inf
    best_h = np.ones(N) / N
    best_score = _score(best_h)
    
    if time.time() < deadline - 20:
        h_freq, score_freq = _freq_optimize()
        if score_freq < best_score:
            best_score = score_freq
            best_h = h_freq.copy()
    
    if time.time() < deadline - 3:
        h_ref, score_ref = _local_refine(best_h, deadline)
        if score_ref < best_score:
            best_score = score_ref
            best_h = h_ref.copy()
    
    return best_h

# EVOLVE-BLOCK-END
