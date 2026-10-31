import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 50

def _score(h):
    """
    Calculate the score: ratio of squared L2 norm to L1 norm times supremum norm of autoconvolution.
    h: array of non-negative heights.
    """
    if np.any(h < -1e-12):
        return -np.inf
    h = np.clip(h, 0, None)
    
    l1 = np.sum(h)
    if l1 == 0:
        return -np.inf
        
    l2_sq = np.sum(h**2)
    
    # Autoconvolution
    # h * h = np.convolve(h, h, mode='full')
    # The supremum norm is the max absolute value.
    # Since h >= 0, autoconv is non-negative.
    # Max is typically at lag 0 (sum h_i^2).
    # But we must check all lags.
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    
    if sup_conv == 0:
        return -np.inf
        
    return l2_sq / (l1 * sup_conv)

def _parametric_heights(params):
    """
    Generate heights from parameters.
    params: [a, b, theta, phi]
    h_i = max(0, a * cos(theta * i + phi) + b)
    """
    a, b, theta, phi = params
    i = np.arange(N)
    h = a * np.cos(theta * i + phi) + b
    h = np.clip(h, 0, None)
    return h

def _objective(params):
    h = _parametric_heights(params)
    return -_score(h)

def construct():
    start_time = time.time()
    deadline = 120.0 # Safety margin from 150s limit
    
    # Define bounds for DE
    # a, b can be any real, but scale invariant.
    # We can fix scale to simplify, e.g., max(h) = 1 or sum(h)=1.
    # But DE can handle scale.
    # Let's bound a, b to reasonable ranges to avoid numerical issues.
    # Since scale invariant, we can normalize h after generation?
    # No, the score is scale invariant, so we don't need to fix scale.
    # But DE needs finite bounds.
    # Let's try a in [0.1, 10], b in [-10, 10], theta in [0, 20], phi in [0, 2*pi]
    
    bounds = [(0.1, 10.0), (-10.0, 10.0), (0.0, 20.0), (0.0, 2 * np.pi)]
    
    best_h = np.ones(N) # Fallback
    best_score = _score(best_h)
    
    try:
        # Global Search with Differential Evolution
        # maxiter and popsize adjusted for time
        result_de = differential_evolution(
            _objective, 
            bounds, 
            maxiter=200, 
            popsize=15, 
            seed=42, 
            tol=1e-6,
            polish=False # We will polish separately
        )
        
        if time.time() - start_time > deadline - 20:
            return best_h.tolist()
            
        best_params = result_de.x
        best_h = _parametric_heights(best_params)
        current_score = _score(best_h)
        
        if current_score > best_score:
            best_score = current_score
            best_h = best_h.copy()
            
        # Local Polish with L-BFGS-B
        # L-BFGS-B requires gradient or uses finite differences.
        # We can use 'BFGS' or 'L-BFGS-B' with finite diff.
        # Let's use L-BFGS-B with finite diff (default).
        
        def _obj_local(params):
            return _objective(params)
            
        res_local = minimize(
            _obj_local, 
            best_params, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if time.time() - start_time > deadline - 5:
            return best_h.tolist()
            
        if res_local.success:
            polished_params = res_local.x
            polished_h = _parametric_heights(polished_params)
            polished_score = _score(polished_h)
            
            if polished_score > best_score:
                best_score = polished_score
                best_h = polished_h.copy()
                
    except Exception as e:
        pass
        
    return best_h.tolist()
