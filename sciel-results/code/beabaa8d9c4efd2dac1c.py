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

        # Phase 1: Generate diverse initial points for global exploration
        # Use a combination of structured grid points and random points
        rng = np.random.default_rng(12345)
        
        initial_points = []
        
        # Add center point
        initial_points.append(np.zeros(d))
        
        # Add axis-aligned points (along each dimension at several levels)
        for dim_idx in range(d):
            for level in [-4.0, -2.0, 2.0, 4.0]:
                pt = np.zeros(d)
                pt[dim_idx] = level
                initial_points.append(pt)
        
        # Add random points
        n_random = 30
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
        
        # Phase 2: Local refinement with L-BFGS-B on top candidates
        # Allocate remaining budget among the best starting points
        remaining_budget = budget - calls
        if remaining_budget <= 0:
            return best_x, best_f
        
        # Run L-BFGS-B on the top 10 candidates
        n_refine = 10
        # Distribute remaining budget among the refinement runs
        # Leave some budget for final polishing
        polish_budget = min(100, remaining_budget // 4)
        refine_budget = remaining_budget - polish_budget
        iter_per_run = max(20, refine_budget // n_refine)
        
        for i in range(min(n_refine, len(sorted_points))):
            if not time_ok() or calls >= budget - polish_budget:
                break
            
            x0 = sorted_points[i]
            f0 = sorted_fits[i]
            
            # Skip if this start is significantly worse than current best
            # f11 can have large offsets, so be careful with this threshold
            if f0 > best_f + 500:
                continue
            
            # Run L-BFGS-B with moderate settings
            try:
                res = minimize(wrapped_func, x0, method='L-BFGS-B',
                               bounds=[(-5, 5)] * d,
                               options={'maxiter': iter_per_run, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass
            
            if not time_ok() or calls >= budget - polish_budget:
                break
        
        # Phase 3: Polish with very tight convergence from the best point
        if not time_ok() or calls >= budget:
            return best_x, best_f
        
        if calls < budget - 10:
            # Use the remaining budget for a high-precision polish
            polish_iters = budget - calls - 10
            try:
                res = minimize(wrapped_func, best_x, method='L-BFGS-B',
                               bounds=[(-5, 5)] * d,
                               options={'maxiter': polish_iters, 'ftol': 1e-16, 'gtol': 1e-12})
                
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
