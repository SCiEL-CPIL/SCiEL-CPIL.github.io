import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400

def compute_value(h):
    """Compute the task value."""
    h_sum = np.sum(h)
    if h_sum == 0 or not np.isfinite(h_sum):
        return float('inf')
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    return abs(2 * N * max_conv / (h_sum ** 2))

def _build_chirp(params):
    """
    Build h from chirp parameters.
    params = [A, f, beta, offset]
    h[n] = offset + A * cos(2*pi*(f*n + beta*n^2/N))
    """
    A = params[0]
    f = params[1]
    beta = params[2]
    offset = params[3]
    n = np.arange(N, dtype=np.float64)
    phase = 2.0 * np.pi * (f * n + beta * n * n / N)
    h = offset + A * np.cos(phase)
    return h

def _chirp_objective(params):
    h = _build_chirp(params)
    s = np.sum(h)
    if abs(s) < 1e-10 or not np.isfinite(s):
        return 1e6
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    return abs(2.0 * N * max_conv / (s * s))

def _optimize_one(bounds, x0, time_limit=30.0):
    """Run differential evolution + local refinement from a starting point."""
    start = time.time()
    
    # Differential evolution
    de_result = differential_evolution(
        _chirp_objective, bounds, seed=42, maxiter=200, tol=1e-6,
        popsize=30, polish=True, init='latinhypercube'
    )
    
    best_params = de_result.x
    best_val = de_result.fun
    
    # Local refinement with Nelder-Mead
    if time.time() - start < time_limit - 5:
        lm_result = minimize(
            _chirp_objective, best_params, method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if lm_result.fun < best_val:
            best_params = lm_result.x
            best_val = lm_result.fun
    
    return best_params, best_val

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)  # Should be 2.0
    
    # Bounds for chirp parameters
    # A: amplitude, f: linear frequency, beta: quadratic (chirp rate), offset: DC
    bounds = [
        (0.1, 2.0),      # A: amplitude
        (0.0, 5.0),      # f: frequency (cycles over N samples)
        (0.0, 10.0),     # beta: chirp rate
        (-0.5, 2.0),     # offset: DC level
    ]
    
    # Try multiple random restarts with different initial points
    np.random.seed(42)
    
    num_restarts = 8
    for i in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Generate random initial point within bounds
        x0 = np.array([
            np.random.uniform(bounds[0][0], bounds[0][1]),
            np.random.uniform(bounds[1][0], bounds[1][1]),
            np.random.uniform(bounds[2][0], bounds[2][1]),
            np.random.uniform(bounds[3][0], bounds[3][1]),
        ])
        
        params_opt, val_opt = _optimize_one(bounds, x0, time_limit=15.0)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = _build_chirp(params_opt)
    
    # Final polish: try to refine the best h directly using subgradient on h
    if time.time() - start_time < time_limit - 10:
        h = best_h.copy()
        h_sum = np.sum(h)
        h = h / h_sum  # normalize to sum 1 for stability
        
        # Simple gradient-based polish on h directly
        eta = 0.01
        for it in range(500):
            if time.time() - start_time > time_limit - 5:
                break
            conv = np.convolve(h, h)
            k_star = np.argmax(conv)
            
            j = np.arange(N)
            idx1 = k_star - j
            idx2 = j - k_star
            mask1 = (idx1 >= 0) & (idx1 < N)
            mask2 = (idx2 >= 0) & (idx2 < N)
            
            g = np.zeros(N)
            g[mask1] += h[idx1[mask1]]
            g[mask2] += h[idx2[mask2]]
            
            h = h - eta * g
            h = h - (np.sum(h) - 1) / N
            
            eta *= 0.999
            
            val = compute_value(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
    
    return best_h
