"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: coarse grid search + Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        best_x = None
        best_f = float('inf')
        evals = 0

        def check_deadline():
            return time.time() > deadline

        def update_best(x, f):
            nonlocal best_x, best_f, evals
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Coarse grid search to find promising region
        # Try a few strategic points first
        candidate_points = [
            np.zeros(n),
            np.full(n, 2.0),
            np.full(n, -2.0),
            np.linspace(-3, 3, n),
            np.array([1.0, -1.0, 1.0, -1.0, 1.0][:n]),
            np.array([-1.0, 1.0, -1.0, 1.0, -1.0][:n]),
        ]
        
        for pt in candidate_points:
            if check_deadline() or evals >= budget:
                break
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            evals += 1
            update_best(pt, f)
        
        # If we have enough budget, do a small grid
        if not check_deadline() and evals < budget - 50:
            # 2-point grid per dimension: -2, 0, 2
            grid_vals = [-2.0, 0.0, 2.0]
            # Only do this if we have plenty of budget
            grid_size = 3 ** n
            if grid_size <= budget - evals - 30:
                for i in range(grid_size):
                    if check_deadline() or evals >= budget - 30:
                        break
                    # Convert index to grid coordinates
                    coords = []
                    temp = i
                    for d in range(n):
                        coords.append(grid_vals[temp % 3])
                        temp //= 3
                    pt = np.array(coords)
                    f = func(pt)
                    evals += 1
                    update_best(pt, f)
        
        # Phase 2: Local refinement using Nelder-Mead from best point
        if check_deadline() or best_x is None:
            return best_x, best_f
        
        remaining = budget - evals
        if remaining < 10:
            return best_x, best_f
        
        # Use Nelder-Mead for local optimization
        try:
            # Set up Nelder-Mead options
            nm_options = {
                'maxiter': remaining - 5,
                'maxfev': remaining - 5,
                'xatol': 1e-6,
                'fatol': 1e-10,
                'adaptive': True
            }
            
            # Wrap func to track evaluations
            def wrapped_func(x):
                nonlocal evals
                if check_deadline() or evals >= budget:
                    return float('inf')
                x = np.clip(x, -5, 5)
                f = func(x)
                evals += 1
                return f
            
            result = minimize(wrapped_func, best_x, method='Nelder-Mead', options=nm_options)
            
            if result.success or result.fun < best_f:
                update_best(result.x, result.fun)
                
        except Exception:
            pass
        
        # Phase 3: If still have budget, do a few more random local perturbations
        if not check_deadline() and evals < budget and best_x is not None:
            rng = np.random.default_rng(42)
            sigma = 0.5
            no_improve_count = 0
            
            while evals < budget and not check_deadline():
                x_new = best_x + sigma * rng.standard_normal(n)
                x_new = np.clip(x_new, -5, 5)
                f_new = func(x_new)
                evals += 1
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                    no_improve_count = 0
                    sigma = min(sigma * 1.2, 3.0)
                else:
                    no_improve_count += 1
                    sigma *= 0.8
                    if sigma < 1e-4:
                        break
                
                if no_improve_count > 50:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
