"""
Rewritten program for third autocorrelation inequality optimization.
Strategy: Minimize a composite objective consisting of a soft-max approximation 
of the peak autocorrelation and a penalty on the L2 norm of the autocorrelation 
(spectral flattening). This encourages the autocorrelation to be both low in 
peak and spread out (flat), which is structurally different from just minimizing 
the peak.
"""
import numpy as np
import time
from scipy.fft import fft, ifft
from scipy.optimize import minimize

N = 400

def _eval_value(h):
    """Compute the exact objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    # Use FFT for speed, but np.convolve is also fine for N=400
    # np.convolve is exact and simple
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
    # Shift to prevent overflow
    exp_vals = np.exp(beta * (x - m))
    sm = m + np.log(np.sum(exp_vals)) / beta
    return sm

def _objective_composite(h, beta, alpha):
    """
    Composite objective: soft_max(conv) + alpha * mean(conv^2).
    h is normalized to sum=1 internally to handle scale invariance.
    """
    if not np.all(np.isfinite(h)):
        return 1e10
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    
    # Normalize to sum=1
    h_norm = h / s
    
    conv = _convolve_fft(h_norm)
    
    # Peak component
    peak_val = _soft_max(conv, beta)
    
    # Energy component (L2 norm squared of conv)
    # This penalizes concentration of energy in the autocorrelation
    energy_val = np.mean(conv * conv)
    
    return peak_val + alpha * energy_val

def construct():
    """
    Returns a sequence of 400 signed step heights.
    """
    deadline = time.time() + 140.0  # Safety margin
    
    # Start with all-ones
    best_h = np.ones(N)
    best_val = _eval_value(best_h)  # 2.0
    
    # Phase 1: Parametric initializations to find a good starting point
    # We try to find structures that already have lower values
    
    # Family 1: Cosine waves (try to create destructive interference in peaks)
    for m in range(1, 30):
        if time.time() > deadline - 80:
            break
        for a in np.linspace(-0.9, 0.9, 19):
            h = 1.0 + a * np.cos(2.0 * np.pi * m * np.arange(N) / N)
            v = _eval_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
                
    # Family 2: Random restarts with local search to break symmetry
    rng = np.random.RandomState(42)
    for _ in range(20):
        if time.time() > deadline - 80:
            break
        h_init = rng.randn(N)
        # Ensure sum is not zero
        if abs(np.sum(h_init)) < 1e-6:
            h_init[0] += 1.0
        h_init = h_init / np.sum(h_init) # Normalize to sum 1
        v = _eval_value(h_init)
        if v < best_val:
            best_val = v
            best_h = h_init.copy()

    # Phase 2: Optimization with Composite Objective
    # We use L-BFGS-B on the composite objective
    
    # Parameters for the composite objective
    # beta controls sharpness of max approximation
    # alpha controls weight of energy penalty
    
    betas = [10.0, 50.0, 200.0, 1000.0]
    alphas = [0.1, 0.01, 0.001]
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
        
        for alpha in alphas:
            if time.time() > deadline - 20:
                break
            
            def obj_func(h):
                return _objective_composite(h, beta, alpha)
            
            # Try a few restarts
            for trial in range(3):
                if time.time() > deadline - 20:
                    break
                
                if trial == 0:
                    h_init = best_h.copy()
                else:
                    # Perturbed restart
                    h_init = best_h + 0.1 * rng.randn(N)
                    # Re-normalize
                    s = np.sum(h_init)
                    if abs(s) > 1e-6:
                        h_init = h_init / s
                    else:
                        h_init = best_h.copy()
                
                try:
                    result = minimize(
                        obj_func,
                        h_init,
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    
                    h_opt = result.x
                    if np.all(np.isfinite(h_opt)):
                        v_opt = _eval_value(h_opt)
                        if v_opt < best_val:
                            best_val = v_opt
                            best_h = h_opt.copy()
                except Exception:
                    continue

    # Phase 3: High-precision polish
    # Use very high beta to approximate max closely, and small alpha
    if time.time() < deadline - 10:
        for beta in [2000.0, 5000.0]:
            if time.time() > deadline - 5:
                break
            
            def obj_polish(h):
                return _objective_composite(h, beta, 0.0001)
            
            try:
                result = minimize(
                    obj_polish,
                    best_h.copy(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                h_opt = result.x
                if np.all(np.isfinite(h_opt)):
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
            except Exception:
                continue

    # Phase 4: Coordinate descent / Local search on the exact objective
    # This helps escape local minima of the soft-max approximation
    if time.time() < deadline - 5:
        h = best_h.copy()
        step = 0.05
        for iteration in range(50):
            if time.time() > deadline - 2:
                break
            improved = False
            # Randomly select indices to perturb
            indices = rng.choice(N, size=20, replace=False)
            for i in indices:
                if time.time() > deadline - 2:
                    break
                old = h[i]
                for delta in [step, -step]:
                    h[i] = old + delta
                    v = _eval_value(h)
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()
                        improved = True
                    else:
                        h[i] = old
            if not improved:
                step *= 0.5
                if step < 0.001:
                    break

    # Ensure valid output
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
