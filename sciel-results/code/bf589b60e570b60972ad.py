"""
Strategy: Randomized Coordinate Descent with Random Restarts and Best-Keeping

Instead of using a structured signal (Zadoff-Chu) and a multi-parameter optimizer (Powell),
we treat the problem as a direct 400-dimensional optimization task but use a simpler,
more robust stochastic local search: Randomized Coordinate Descent (RCD) with random restarts.

Key differences:
1. No structured initialization (no Zadoff-Chu, no chirps). We start from random Gaussian noise.
2. No complex multi-method optimization. We use a simple, fast coordinate descent step where
   we pick one random coordinate, perform a 1D line search (via golden section or ternary search
   on the non-smooth objective), and update.
3. Multiple random restarts to escape local minima, keeping the best result found.
4. The objective is non-smooth due to the max operator, but 1D line searches can handle this
   reasonably well if the function is unimodal along the coordinate direction (which is often
   the case for such quadratic-like forms).

This approach is more "brute force" but less dependent on the quality of the initial structure,
and may find different basins of attraction.
"""

import numpy as np
import time
import random

N = 400
_DEADLINE = 145.0

def _evaluate(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    m = np.max(conv)
    return abs(2.0 * N * m / (s * s))

def _line_search_1d(h, idx, obj_func, x0=-10.0, x1=10.0, tol=1e-4, max_iter=50):
    """
    Perform a 1D line search along coordinate idx of h.
    We use a simple golden section search, but since the function is non-smooth,
    we might use a ternary search or just sample a few points.
    Actually, for non-smooth functions, gradient-free methods like Nelder-Mead on 1D
    or simple sampling can work. Let's try a simple approach:
    Sample a grid of values for h[idx] and pick the best.
    This is robust but might be slow. Let's optimize.
    
    Alternative: Use scipy.optimize.minimize_scalar with bounded=True.
    """
    from scipy.optimize import minimize_scalar
    
    def f_1d(x):
        h[idx] = x
        return obj_func(h)
    
    # Save original value
    orig_val = h[idx]
    
    try:
        res = minimize_scalar(f_1d, bounds=(x0, x1), method='bounded', options={'maxiter': 20, 'xatol': tol})
        best_x = res.x
        best_val = res.fun
    except Exception:
        best_x = orig_val
        best_val = obj_func(h)
    
    h[idx] = best_x
    return best_val

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Set random seed for reproducibility? No, let it be random for exploration.
    # But we want to ensure we find a good solution.
    
    num_restarts = 20  # We will do up to 20 restarts, or until time runs out
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize with random Gaussian noise
        # Scale matters? The objective is scale-invariant, so we can use any scale.
        # Let's use standard normal.
        h = np.random.randn(N)
        
        # Ensure sum is not zero (highly unlikely, but check)
        if np.sum(h) == 0:
            h += 1e-6
        
        # Perform coordinate descent
        # We will do a fixed number of iterations per restart, or until time runs out.
        # Let's say 1000 iterations per restart, but check time.
        max_iters = 1000
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick a random coordinate
            idx = np.random.randint(N)
            
            # Perform line search
            # The range for the line search should be adaptive or wide enough.
            # Since the objective is scale-invariant, the "optimal" value for a coordinate
            # depends on the other coordinates. Let's use a wide range.
            # Actually, let's just use the current value as center and search +/- some delta.
            # Or just use a fixed wide range like [-10, 10].
            
            # To speed up, we can evaluate the objective at a few points instead of full minimize_scalar.
            # Let's try a simple 3-point search or just update if a random perturbation helps.
            # Actually, minimize_scalar is robust. Let's use it but limit iterations.
            
            val = _line_search_1d(h, idx, _evaluate, x0=-10.0, x1=10.0, tol=1e-3, max_iter=10)
            
            # Check if we improved the global best
            if val < best_val:
                best_val = val
                best_h = h.copy()
        
        # If we found a good solution, we can stop early
        if best_val < 1.01:
            break
    
    # Final check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        return np.ones(N)
        
    return best_h
