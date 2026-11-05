"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fftpack import dct, idct

def _fft_autoconv(h):
    """Compute full autoconvolution using FFT for speed."""
    n = len(h)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fh = np.fft.rfft(h, size)
    conv = np.fft.irfft(fh * fh, size)[:2 * n - 1]
    return conv

def _compute_value(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = _fft_autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _grad_autoconv(w, h):
    """
    Gradient of sum_j w_j * conv_j w.r.t. h.
    conv = full autoconvolution of h (length 2N-1).
    d(conv_j)/d(h_l) = 2 * h[j-l] if 0 <= j-l < N, else 0.
    So grad_h[l] = 2 * sum_j w_j * h[j-l] (for valid j-l).
    This is 2 * cross_correlation(w, h)[l] where cross_corr[l] = sum_j w[j] * h[j-l].
    Using FFT: cross_corr = ifft(fft(w) * conj(fft(h)))
    """
    L = 2 * N
    w_pad = np.zeros(L)
    w_pad[:len(w)] = w
    h_pad = np.zeros(L)
    h_pad[:N] = h
    
    W = np.fft.fft(w_pad, n=L)
    H = np.fft.fft(h_pad, n=L)
    
    # Cross-correlation: sum_j w[j] * h[j-l]
    # This is ifft(W * conj(H)).real
    cross = np.fft.ifft(W * np.conj(H)).real
    grad_h = 2.0 * cross[:N]
    return grad_h

def _soft_max_and_weights(conv, beta):
    """Compute soft-max and its weights for the autoconvolution."""
    m = np.max(conv)
    if m <= 0:
        return m, np.zeros_like(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    if sum_exp < 1e-300:
        return m, np.zeros_like(conv)
    weights = exp_shifted / sum_exp
    soft_max = m + np.log(sum_exp) / beta
    return soft_max, weights

def _objective_soft(h, beta=50.0):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _fft_autoconv(h)
    soft_max, _ = _soft_max_and_weights(conv, beta)
    return 2.0 * N * soft_max / (s * s)

def _objective_soft_grad(h, beta=50.0):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    
    conv = _fft_autoconv(h)
    soft_max, weights = _soft_max_and_weights(conv, beta)
    
    if np.max(weights) < 1e-15:
        return np.zeros_like(h)
    
    grad_sm = _grad_autoconv(weights, h)
    
    # f = 2*N*soft_max/s^2
    # df/dh_i = 2*N * (grad_sm_i * s^2 - soft_max * 2*s) / s^4
    #         = 2*N * (grad_sm_i * s - 2*soft_max) / s^3
    return 2.0 * N * (grad_sm * s - 2.0 * soft_max) / (s * s * s)

def _spectral_penalty(h, lam=0.1):
    """Penalty for spectral concentration (L4 norm of DCT coefficients)."""
    dct_coeffs = dct(h, type=2, norm='ortho')
    return lam * np.sum(np.abs(dct_coeffs)**4) / N

def _spectral_penalty_grad(h, lam=0.1):
    """Gradient of spectral penalty."""
    dct_coeffs = dct(h, type=2, norm='ortho')
    grad = 4.0 * lam / N * idct(dct_coeffs**3, type=2, norm='ortho')
    return grad

def _combined_obj(h, beta=50.0, lam=0.05):
    return _objective_soft(h, beta) + _spectral_penalty(h, lam)

def _combined_grad(h, beta=50.0, lam=0.05):
    return _objective_soft_grad(h, beta) + _spectral_penalty_grad(h, lam)

def _refine(h, deadline):
    """Refine a candidate h using a sequence of optimization phases."""
    h_curr = h.copy()
    
    # Phase 1: Moderate beta, spectral flattening
    if time.time() < deadline - 15.0:
        try:
            res = minimize(_combined_obj, h_curr, method='L-BFGS-B',
                           args=(30.0, 0.1), jac=_combined_grad,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-7})
            h_curr = res.x
        except Exception:
            pass
    
    # Phase 2: Higher beta, less spectral penalty (focus on peak)
    if time.time() < deadline - 10.0:
        try:
            res = minimize(_combined_obj, h_curr, method='L-BFGS-B',
                           args=(80.0, 0.01), jac=_combined_grad,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9})
            h_curr = res.x
        except Exception:
            pass
    
    # Phase 3: Pure peak polishing with very high beta
    if time.time() < deadline - 5.0:
        try:
            res = minimize(_objective_soft, h_curr, method='L-BFGS-B',
                           args=(150.0,), jac=_objective_soft_grad,
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            h_curr = res.x
        except Exception:
            pass
            
    return h_curr

def _simulated_annealing_binary(h_init, deadline, rng):
    """
    Binary Simulated Annealing to find low-peak sequences.
    Represents h as binary {-1, 1} initially, then allows small perturbations.
    """
    # Start with a binary-like sequence derived from h_init
    h = np.sign(h_init)
    # Ensure sum is not zero
    if np.sum(h) == 0:
        h[0] = 1.0
        
    best_h = h.copy()
    best_val = _compute_value(h)
    
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.995
    
    steps = 0
    max_steps = 5000
    
    while time.time() < deadline - 5.0 and steps < max_steps:
        T = T_start * (T_end / T_start) ** (steps / max_steps)
        
        # Pick a random index to flip
        i = rng.integers(0, N)
        old_val = h[i]
        h[i] = -h[i]
        
        new_val = _compute_value(h)
        
        delta = new_val - best_val
        if delta < 0 or rng.random() < np.exp(-delta / (T + 1e-10)):
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            h[i] = old_val
            
        steps += 1
        
    return best_h, best_val

def _unitary_spectral_init(rng, num_impulses=50):
    """
    Generate h as the real part of a complex vector from a random unitary transformation
    of a sparse impulse train. This enforces a flat power spectrum via Parseval's theorem.
    """
    # Create a sparse impulse train in frequency domain
    freqs = np.zeros(N, dtype=complex)
    indices = rng.choice(N, size=num_impulses, replace=False)
    for idx in indices:
        freqs[idx] = rng.standard_normal() + 1j * rng.standard_normal()
    
    # Apply random unitary transformation (DFT is unitary up to scaling)
    # h = ifft(freqs)
    h = np.fft.ifft(freqs).real
    
    # Normalize to have reasonable magnitude
    h = h / (np.std(h) + 1e-10)
    
    # Add a DC offset to ensure sum is not zero
    h += 0.5 * rng.standard_normal()
    
    return h

def _unitary_spectral_optimize(h0, deadline, rng):
    """
    Optimize the spectral representation of h by adjusting the DFT coefficients.
    This works in the frequency domain to control the autoconvolution peak.
    """
    # Get DFT of h0
    H = np.fft.rfft(h0)
    
    # Parameterize: optimize the magnitudes and phases of the DFT coefficients
    # Keep DC component fixed to control sum(h)
    dc = H[0].real
    
    # Initialize phases and magnitudes
    magnitudes = np.abs(H).copy()
    phases = np.angle(H).copy()
    
    # Fix DC magnitude
    magnitudes[0] = abs(dc)
    
    def _reconstruct(mags, phs):
        H_new = mags * np.exp(1j * phs)
        # Ensure DC is real and positive
        H_new[0] = abs(dc)
        h_new = np.fft.irfft(H_new, n=N)
        return h_new
    
    def _spectral_objective(params):
        mags = params[:N//2+1]
        phs = params[N//2+1:]
        h = _reconstruct(mags, phs)
        return _objective_soft(h, beta=50.0)
    
    # Initial parameters
    init_params = np.concatenate([magnitudes, phases])
    
    try:
        res = minimize(_spectral_objective, init_params, method='L-BFGS-B',
                       options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.fun < _compute_value(h0):
            mags = res.x[:N//2+1]
            phs = res.x[N//2+1:]
            h_new = _reconstruct(mags, phs)
            return h_new
    except Exception:
        pass
    
    return h0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    i = np.arange(N)
    
    # 1. Seed
    initializations.append(np.ones(N))
    
    # 2. Low-frequency sinusoids with DC offset
    for k in [1, 2, 3, 4]:
        for a in [0.5, 1.0, 2.0]:
            h0 = a + np.sin(2*np.pi*k*i/N)
            initializations.append(h0)
            
    # 3. Comb patterns
    for period in [2, 3, 4, 5, 8, 10]:
        h0 = np.where((i % period) < period // 2, 1.0, -1.0)
        h0 += 0.1 * rng.standard_normal(N)
        if abs(np.sum(h0)) > 1.0:
            initializations.append(h0)
            
    # 4. Random smooth functions
    for _ in range(5):
        h0 = np.sin(2*np.pi*rng.uniform(0.1, 2.0)*i/N + rng.uniform(0, 2*np.pi))
        h0 += 1.0
        initializations.append(h0)
    
    # 5. Unitary spectral initializations (NEW)
    for num_imp in [20, 50, 100]:
        for _ in range(3):
            h0 = _unitary_spectral_init(rng, num_impulses=num_imp)
            if abs(np.sum(h0)) > 0.1:
                initializations.append(h0)
    
    # Optimize each initialization with gradient descent
    for h0 in initializations:
        if time.time() > deadline - 30.0:
            break
            
        h_refined = _refine(h0, deadline)
        val = _compute_value(h_refined)
        if val < best_val:
            best_val = val
            best_h = h_refined.copy()
            
    # Try unitary spectral optimization on the best continuous solution
    if time.time() < deadline - 20.0:
        h_spec = _unitary_spectral_optimize(best_h, deadline, rng)
        val_spec = _compute_value(h_spec)
        if val_spec < best_val:
            best_val = val_spec
            best_h = h_spec.copy()
            # Refine further
            if time.time() < deadline - 10.0:
                h_refined = _refine(best_h, deadline)
                val_refined = _compute_value(h_refined)
                if val_refined < best_val:
                    best_val = val_refined
                    best_h = h_refined.copy()
    
    # Try binary Simulated Annealing on the best continuous solution
    if time.time() < deadline - 10.0:
        h_sa, val_sa = _simulated_annealing_binary(best_h, deadline, rng)
        if val_sa < best_val:
            best_val = val_sa
            best_h = h_sa.copy()
            
            # Refine the binary result with continuous optimization
            if time.time() < deadline - 5.0:
                h_refined = _refine(best_h, deadline)
                val_refined = _compute_value(h_refined)
                if val_refined < best_val:
                    best_val = val_refined
                    best_h = h_refined.copy()

    # Final global check and return
    if time.time() < deadline - 1.0:
        h_final = _refine(best_h, deadline)
        val_final = _compute_value(h_final)
        if val_final < best_val:
            best_h = h_final
            
    s_final = np.sum(best_h)
    if abs(s_final) < 1e-10:
        best_h = np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
