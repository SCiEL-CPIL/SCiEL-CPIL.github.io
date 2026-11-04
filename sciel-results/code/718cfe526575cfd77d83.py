import numpy as np
from scipy.optimize import minimize
import time

N = 600
DELTA = (1.0 / 2.0) / N  # interval width = 1/1200

def _compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    Score = max(f*f) / (integral f)^2
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-9):
        return 1e10
    h = np.maximum(h, 0)
    
    integral = np.sum(h) * DELTA
    if integral <= 1e-10:
        return 1e10
        
    # For non-negative step functions, the autoconvolution maximum occurs at t=0
    # (f*f)(0) = sum(h^2) * DELTA
    max_ac = np.sum(h**2) * DELTA
    
    return max_ac / (integral ** 2)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Strategy: Represent h(x) = |sum_{k=0}^{K} c_k e^{i 4 pi k x}|^2
    # This ensures h(x) >= 0 by construction.
    
    n_modes = 15
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    
    # Precompute exponentials for speed
    exp_terms = np.zeros((n_modes, N), dtype=complex)
    for k in range(n_modes):
        freq = 4.0 * np.pi * k
        exp_terms[k] = np.exp(1j * freq * x)
    
    def params_to_h(params):
        # params: [re(c0), im(c0), re(c1), im(c1), ...]
        c = np.zeros(n_modes, dtype=complex)
        for i in range(n_modes):
            re = params[2*i]
            im = params[2*i+1] if i > 0 else 0.0
            c[i] = re + 1j*im
        
        # h(x) = |sum c_k exp(ikx)|^2
        h_complex = np.zeros(N, dtype=complex)
        for k in range(n_modes):
            h_complex += c[k] * exp_terms[k]
        
        h = np.abs(h_complex)**2
        return h

    def objective(params):
        h = params_to_h(params)
        return _compute_score(h)
    
    # Initial guess: Uniform (c[0]=1, others 0)
    best_h = np.ones(N)
    best_score = _compute_score(best_h)
    
    n_params = 2 * n_modes
    rng = np.random.default_rng(42)
    
    # Phase 1: Coarse multi-start with moderate iterations
    n_starts = 8
    for s in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        x0 = np.zeros(n_params)
        x0[0] = 1.0
        
        if s == 0:
            # Start from uniform
            pass
        else:
            # Random perturbations with varying scales
            scale = 0.05 + 0.05 * s
            x0[2:] += rng.normal(0, scale, size=n_params-2)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-7, 'gtol': 1e-5})
            if res.fun < best_score:
                best_score = res.fun
                best_h = params_to_h(res.x)
        except Exception:
            pass
    
    # Phase 2: Fine polish from best result with tighter tolerances
    if time.time() < deadline - 15:
        x0 = np.zeros(n_params)
        # Reconstruct params from best_h by least squares fit
        # h(x) = |sum c_k exp(ikx)|^2, so we need to find c such that |sum c_k exp(ikx)|^2 ≈ best_h
        # This is tricky; instead, use the last optimizer's x if available
        # For now, try polishing from a few nearby points
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            if attempt == 0:
                # Try the uniform start again with more iterations
                x0_polish = np.zeros(n_params)
                x0_polish[0] = 1.0
            else:
                # Small random perturbation around uniform
                x0_polish = np.zeros(n_params)
                x0_polish[0] = 1.0
                x0_polish[2:] += rng.normal(0, 0.01, size=n_params-2)
            
            try:
                res = minimize(objective, x0_polish, method='L-BFGS-B', 
                               options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.fun < best_score:
                    best_score = res.fun
                    best_h = params_to_h(res.x)
            except Exception:
                pass
    
    # Final check
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-10:
        best_h = np.ones(N)
        
    return list(best_h)
