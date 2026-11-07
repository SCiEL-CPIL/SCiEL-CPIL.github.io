"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start L-BFGS-B + Nelder-Mead + (1+1)-ES for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x, best_f = x.copy(), f

        def wrapped_func(x):
            """Wrapper that counts calls and tracks best."""
            nonlocal calls_made
            if not time_ok() or calls_made >= budget:
                return 1e10
            fx = func(x)
            calls_made += 1
            try_update(x, fx)
            return fx

        # Phase 1: Generate diverse seeds - mix of random and structured
        n_explore = 20
        seeds = []
        
        # Add origin and some structured points
        structured_points = [
            np.zeros(dim),
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
        ]
        
        for i in range(n_explore):
            if not time_ok() or calls_made >= budget:
                break
            if i < len(structured_points):
                x = structured_points[i].copy()
            else:
                x = rng.uniform(-5, 5, dim)
            fx = wrapped_func(x)
            seeds.append((x.copy(), fx))
            
            if calls_made >= budget:
                break

        if not seeds:
            x = np.zeros(dim)
            fx = func(x)
            calls_made += 1
            try_update(x, fx)
            return best_x, best_f

        # Sort seeds by fitness
        seeds.sort(key=lambda s: s[1])
        
        # Phase 2: L-BFGS-B Local Optimization from best seeds
        n_local_runs = 6
        remaining_budget = budget - calls_made
        calls_per_local = max(15, remaining_budget // (n_local_runs + 1))
        
        for i in range(min(n_local_runs, len(seeds))):
            if not time_ok() or calls_made >= budget:
                break
            
            start_x, start_f = seeds[i]
            
            try:
                res = minimize(wrapped_func, start_x, method='L-BFGS-B', 
                               bounds=[(-5, 5)] * dim,
                               options={'maxiter': 100, 'maxfun': calls_per_local,
                                        'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
                if res.fun < best_f:
                    best_x = res.x.copy()
                    best_f = res.fun
            except Exception:
                pass

        # Phase 3: Nelder-Mead refinement from best point if budget remains
        if best_x is not None and calls_made < budget - 30 and time_ok():
            remaining = budget - calls_made
            if remaining > 40:
                try:
                    nm_budget = min(remaining - 10, 80)
                    res = minimize(wrapped_func, best_x, method='Nelder-Mead',
                                   options={'maxiter': nm_budget, 'xatol': 1e-10, 'fatol': 1e-12})
                    if res.fun < best_f:
                        best_x = res.x.copy()
                        best_f = res.fun
                except Exception:
                    pass

        # Phase 4: Polishing with (1+1)-ES
        if best_x is not None and calls_made < budget and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            stall = 0
            max_stall = 40
            
            while calls_made < budget and time_ok():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = wrapped_func(y)
                
                if fy < fx:
                    x, fx = y.copy(), fy
                    sigma = min(sigma * 1.1, 0.5)
                    stall = 0
                else:
                    sigma = max(sigma * 0.9, 1e-8)
                    stall += 1
                    if stall > max_stall:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
