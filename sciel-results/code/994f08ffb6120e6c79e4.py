"""Combined program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import minimize

def _eval_value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _convolve_fft(h):
    """Compute full autoconvolution using FFT."""
    n_conv = 2 * len(h) - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = fft(h, n=n_fft)
    conv = ifft(H * H, n=n_fft).real[:n_conv]
    return conv

def _soft_max(x, beta):
    """Numerically stable soft maximum."""
    m = np.max(x)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp(beta * (x - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    return sm

def _objective_val(h, beta):
    """
    Objective function value.
    """
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _convolve_fft(h)
    sm = _soft_max(conv, beta)
    return 2.0 * N * sm / (s * s)

def _objective_grad(h, beta):
    """
    Analytic gradient of the objective function.
    """
    if not np.all(np.isfinite(h)):
        return np.zeros(N)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros(N)
    
    n_conv = 2 * N - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
        
    H = fft(h, n=n_fft)
    C_full = ifft(H * H, n=n_fft).real
    C = C_full[:n_conv]
    
    # Calculate soft-max weights
    m = np.max(C)
    exp_vals = np.exp(beta * (C - m))
    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        return np.zeros(N)
    w = exp_vals / sum_exp
    
    # Pad w to n_fft
    w_padded = np.zeros(n_fft)
    w_padded[:n_conv] = w
    
    # Gradient of soft-max M w.r.t h
    # G = 2 * ifft(fft(w_padded) * conj(fft(h)))
    W = fft(w_padded, n=n_fft)
    grad_M = 2.0 * ifft(W * np.conj(H), n=n_fft).real[:N]
    
    # Objective J = 2N * M / s^2
    # dJ/dh = (2N / s^2) * dM/dh - (2N * M / s^3) * ds/dh
    # ds/dh = 1
    M = m + np.log(sum_exp) / beta
    val_M = M
    val_s2 = s * s
    
    grad_J = (2.0 * N / val_s2) * grad_M - (2.0 * N * val_M / (val_s2 * s)) * 1.0
    
    return grad_J

def _spectral_flatness_obj(c_ac, c_dc, weight=1.0):
    """
    Objective combining spectral flatness (L4 norm of DCT coeffs) 
    with a small soft-max term to guide toward low peak.
    Minimizing L4 norm of DCT coefficients flattens the power spectrum,
    which reduces the peak of the autocorrelation.
    """
    from scipy.fft import dct, idct
    
    c_full = np.concatenate([[c_dc], c_ac])
    h_recon = idct(c_full, type=2, norm='ortho')
    
    if not np.all(np.isfinite(h_recon)):
        return 1e10
    
    s = np.sum(h_recon)
    if abs(s) < 1e-10:
        return 1e10
    
    # Normalize to have sum = N
    h_norm = h_recon * (N / s)
    
    # Compute DCT coefficients of normalized signal
    c_norm = dct(h_norm, type=2, norm='ortho')
    
    # L4 norm of AC components (proxy for spectral flatness)
    # Minimizing this flattens the spectrum
    l4_energy = np.sum(np.abs(c_norm[1:])**4)
    
    # Also compute soft-max for direct peak guidance
    conv = _convolve_fft(h_norm)
    sm = _soft_max(conv, 50.0)
    peak_term = abs(2.0 * N * sm / (N * N))
    
    # Combined objective: weight on spectral flatness + peak term
    return weight * l4_energy / N**4 + (1.0 - weight) * peak_term

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses a hybrid approach:
    1. Parametric initialization
    2. DCT-domain spectral flattening
    3. Time-domain soft-max optimization
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Phase 1: Parametric initializations
    # 1. Sine waves
    for m in range(1, 11):
        if time.time() > deadline - 60:
            break
        for a in np.linspace(-0.9, 0.9, 19):
            h = 1.0 + a * np.sin(2.0 * np.pi * m * np.arange(N) / N)
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
                
    # 2. Random initializations
    rng = np.random.RandomState(42)
    for _ in range(20):
        if time.time() > deadline - 60:
            break
        h = rng.randn(N)
        # Add some DC to avoid sum=0
        h += 0.1 * np.mean(h) 
        # Normalize sum to 1 to help optimization
        s = np.sum(h)
        if abs(s) > 1e-6:
            h_norm = h / s
            v = _eval_value(h_norm)
            if v < best_val:
                best_val = v
                best_h = h_norm.copy()

    # Phase 2: DCT-domain optimization with spectral flattening
    from scipy.fft import dct, idct
    
    def _h_to_dct(h):
        """Convert time-domain signal to DCT-II coefficients."""
        return dct(h, type=2, norm='ortho')
    
    def _dct_to_h(coeffs):
        """Convert DCT-II coefficients to time-domain signal."""
        return idct(coeffs, type=2, norm='ortho')
    
    c_best = _h_to_dct(best_h)
    
    # Phase 2a: Spectral flattening - minimize L4 norm of DCT coefficients
    # This flattens the power spectrum, reducing peak-to-average ratio
    if time.time() < deadline - 70:
        c_dc_flat = c_best[0].copy()
        
        for weight in [0.8, 0.6, 0.4, 0.2]:
            if time.time() > deadline - 70:
                break
            
            def obj_flat(c_ac, weight=weight):
                return _spectral_flatness_obj(c_ac, c_dc_flat, weight)
            
            try:
                result = minimize(
                    obj_flat,
                    c_best[1:].copy(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                c_ac_opt = result.x
                c_full_opt = np.concatenate([[c_dc_flat], c_ac_opt])
                h_opt = _dct_to_h(c_full_opt)
                
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        c_best = _h_to_dct(best_h)
            except Exception:
                continue

    # Phase 3: Optimization with soft-max surrogate and analytic gradient
    # We will run several optimization rounds with increasing beta
    betas = [1.0, 5.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    # We keep the best_h found so far as the starting point for the next beta
    current_h = best_h.copy()
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        # Define objective and gradient with current beta
        def obj(h):
            return _objective_val(h, beta)
            
        def grad(h):
            return _objective_grad(h, beta)
            
        # Try a few restarts from current best and random perturbations
        for trial in range(3):
            if time.time() > deadline - 5:
                break
                
            if trial == 0:
                h_init = current_h.copy()
            else:
                # Perturb the best solution
                noise_scale = 0.1 / (beta + 1.0)
                h_init = current_h + noise_scale * rng.randn(N)
                # Keep sum similar
                h_init -= np.mean(h_init)
                h_init += np.mean(current_h)
                
            try:
                # Use L-BFGS-B with analytic gradient
                result = minimize(
                    obj,
                    h_init,
                    jac=grad,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                h_opt = result.x
                if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
                        current_h = h_opt.copy()
            except Exception:
                continue

    # Phase 4: Final polish with very high beta
    if time.time() < deadline - 5:
        beta = 1000.0
        def obj_polish(h):
            return _objective_val(h, beta)
            
        def grad_polish(h):
            return _objective_grad(h, beta)
            
        try:
            result = minimize(
                obj_polish,
                best_h.copy(),
                jac=grad_polish,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_opt = result.x
            if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-10:
                v_opt = _eval_value(h_opt)
                if v_opt < best_val:
                    best_val = v_opt
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Ensure finite and sum != 0
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
