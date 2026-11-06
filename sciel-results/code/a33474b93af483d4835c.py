import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # Objective function
    def neg_log_score(x):
        if time.time() > deadline:
            return 0.0 # Or some penalty, but DE will stop anyway if we check externally
        pts = x.reshape(8, 2)
        # Compute pairwise distances
        # Use broadcasting
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        # Zero out diagonal
        np.fill_diagonal(dists, 0.0)
        # Get unique pairs (i < j)
        iu = np.triu_indices(8, k=1)
        d_pairs = dists[iu]
        
        # Check for distinct points
        if np.any(d_pairs < 1e-10):
            return 1e10 # Penalty
            
        D = np.max(d_pairs)
        if D < 1e-10:
            return 1e10
            
        # Log product - 8 * Log(D)
        # Sum of logs
        log_prod = np.sum(np.log(d_pairs))
        val = log_prod - 8.0 * np.log(D)
        return -val # Minimize negative log score

    # Initial bounds
    bounds = [(-10.0, 10.0)] * 16
    
    # Differential Evolution
    # We need to be careful with the time limit.
    # DE can be slow. Let's set maxiter and check time in a callback or just run a fixed amount.
    # A better approach: run DE with a time limit via `callback` or just set conservative parameters.
    
    best_result = None
    try:
        # Use a smaller popsize and maxiter to ensure we finish in time
        # 16D problem. Popsize 15 -> 240 points. 
        # Let's try popsize=10, maxiter=500.
        # We will interrupt if time is exceeded.
        
        def callback(xk, convergence=True):
            return time.time() > deadline

        res = differential_evolution(
            neg_log_score, 
            bounds, 
            maxiter=2000, 
            popsize=15, 
            tol=1e-8, 
            callback=callback,
            seed=42,
            polish=False # We will polish separately
        )
        best_result = res.x
    except Exception as e:
        pass

    # If DE didn't find anything or timed out early, best_result might be None or bad
    if best_result is None:
        # Fallback to random or regular octagon
        # Let's just use regular octagon if we failed completely
        return _regular_octagon()

    # Local Polish with Nelder-Mead
    if time.time() < deadline - 5.0:
        try:
            res_nm = minimize(
                neg_log_score, 
                best_result, 
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000, 'maxfev': 1000}
            )
            if res_nm.fun < neg_log_score(best_result):
                best_result = res_nm.x
        except Exception:
            pass

    # Final Polish
    if time.time() < deadline - 1.0:
        try:
            # One more polish with tighter tolerance
            res_nm2 = minimize(
                neg_log_score, 
                best_result, 
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 500, 'maxfev': 500}
            )
            if res_nm2.fun < neg_log_score(best_result):
                best_result = res_nm2.x
        except Exception:
            pass

    pts = best_result.reshape(8, 2)
    
    # Scale to diameter 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    D = np.max(dists)
    if D > 0:
        pts = pts * (2.0 / D)
        
    # Convert to list of lists of floats
    return pts.tolist()

def _regular_octagon():
    from math import cos, sin, pi
    return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
