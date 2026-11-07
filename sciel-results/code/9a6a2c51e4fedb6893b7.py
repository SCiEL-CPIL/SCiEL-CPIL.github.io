"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Deterministic global optimization via L-BFGS-B on a coarse grid."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        d = self.dim
        budget = self.budget
        calls = 0

        def time_ok():
            return time.time() < deadline

        def wrapped_func(x):
            nonlocal calls
            if calls >= budget or not time_ok():
                return 1e10
            calls += 1
            try:
                val = func(x)
                if np.isnan(val) or np.isinf(val):
                    return 1e10
                return val
            except Exception:
                return 1e10

        # Phase 1: Generate diverse initial points
        # Center, Axis points, and Random points
        rng = np.random.default_rng(12345)
        
        initial_points = []
        initial_points.append(np.zeros(d))
        
        # Axis-aligned points (simpler set to save calls)
        for dim_idx in range(d):
            for level in [-4.0, 4.0]:
                pt = np.zeros(d)
                pt[dim_idx] = level
                initial_points.append(pt)
        
        # Random points
        n_random = 10
        for _ in range(n_random):
            initial_points.append(rng.uniform(-5, 5, d))
            
        # Evaluate all initial points
        best_x = initial_points[0].copy()
        best_f = wrapped_func(best_x)
        
        initial_fits = [best_f]
        for i, x in enumerate(initial_points[1:], 1):
            if not time_ok() or calls >= budget:
                break
            fx = wrapped_func(x)
            initial_fits.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Sort initial points by fitness
        sorted_indices = np.argsort(initial_fits)
        sorted_points = [initial_points[i] for i in sorted_indices]
        sorted_fits = [initial_fits[i] for i in sorted_indices]
        
        # Phase 2: Local refinement with L-BFGS-B on all candidates
        # L-BFGS-B is efficient for smooth functions
        n_refine = len(sorted_points)
        
        for i in range(n_refine):
            if not time_ok() or calls >= budget - 10: # Leave 10 for safety
                break
            
            x0 = sorted_points[i]
            f0 = sorted_fits[i]
            
            # Skip if this start is significantly worse than current best
            if f0 > best_f + 50:
                continue
            
            # Run L-BFGS-B
            try:
                res = minimize(wrapped_func, x0, method='L-BFGS-B',
                               bounds=[(-5, 5)] * d,
                               options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass
            
        # Phase 3: If budget remains, do a few more random restarts + L-BFGS-B
        # This helps if the initial grid missed the valley
        while time_ok() and calls < budget - 10:
            # Generate a new random point
            x_new = rng.uniform(-5, 5, d)
            f_new = wrapped_func(x_new)
            
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
            
            # If the new point is good enough, refine it
            if f_new < best_f + 10: 
                try:
                    res = minimize(wrapped_func, x_new, method='L-BFGS-B',
                                   bounds=[(-5, 5)] * d,
                                   options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
                    
                    if res.fun < best_f and np.isfinite(res.fun):
                        best_f = res.fun
                        best_x = res.x.copy()
                        
                except Exception:
                    pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
