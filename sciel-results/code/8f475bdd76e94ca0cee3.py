"""Rewritten program using LHS + Powell + L-BFGS-B hybrid strategy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube
import time

class Optimizer:
    """Hybrid LHS, Powell, and L-BFGS-B optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        
        # Deadline guard: keep 10s margin (300s limit)
        deadline = start_time + 290

        best_x = None
        best_f = np.inf
        call_count = 0

        def wrapped_func(x):
            nonlocal call_count, best_f, best_x
            x = np.asarray(x, dtype=float).flatten()
            if len(x) != dim:
                # Truncate or pad? Truncate is safer if optimizer adds noise, 
                # but optimizers usually maintain dimension.
                # If dimension mismatch, resize to dim.
                if len(x) > dim:
                    x = x[:dim]
                else:
                    x = np.pad(x, (0, dim - len(x)))
            x = np.clip(x, -5.0, 5.0)
            
            call_count += 1
            f_val = func(x)
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
            return f_val

        # 1. Global Search: Latin Hypercube Sampling
        n_global = 100
        sampler = LatinHypercube(d=dim, seed=42)
        lhs_points = sampler.random(n=n_global)
        # Scale from [0, 1] to [-4, 4]
        lhs_points = lhs_points * 8.0 - 4.0

        global_results = []
        for x in lhs_points:
            if time.time() > deadline:
                break
            if call_count >= budget - 100: # Reserve 100 calls for local
                break
            f_val = wrapped_func(x)
            global_results.append((f_val, x))

        if not global_results:
            # Fallback: evaluate origin
            x0 = np.zeros(dim)
            wrapped_func(x0)
            return best_x, best_f

        # Sort results by function value
        global_results.sort(key=lambda item: item[0])
        
        # 2. Local Search: Powell's Method on top candidates
        # Powell's method is effective for Rosenbrock due to its direction set updates
        top_k = 2
        starts = [p for _, p in global_results[:top_k]]
        
        # Allocate iterations based on rank
        max_iters = [30, 15]
        
        for i, x0 in enumerate(starts):
            if time.time() > deadline:
                break
            if call_count >= budget - 20:
                break
            
            mi = max_iters[i] if i < len(max_iters) else 5
            
            try:
                # Powell's method does not use bounds, so we rely on clipping in wrapped_func
                # and the fact that Rosenbrock penalizes large deviations.
                res = minimize(
                    wrapped_func,
                    x0,
                    method='Powell',
                    options={'maxiter': mi, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
            except Exception:
                pass

        # 3. Polish: L-BFGS-B on the best point found
        if time.time() < deadline and call_count < budget - 5:
            try:
                bounds = [(-5.0, 5.0)] * dim
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
