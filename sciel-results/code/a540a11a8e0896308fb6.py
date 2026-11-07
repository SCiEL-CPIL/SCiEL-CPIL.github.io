"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma, restarts, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Phase 1: (1+1)-ES with restarts
        # Use ~70% of budget for exploration with restarts
        es_budget = int(budget * 0.70)
        
        # Restart strategy: if no improvement for a while, restart from best point with larger sigma
        sigma = 3.0
        no_improve = 0
        no_improve_limit = 50
        
        for _ in range(es_budget - 1):
            if not time_ok():
                break
            
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
                no_improve = 0
                sigma = min(sigma * 1.15, 5.0)
            else:
                no_improve += 1
                sigma = max(sigma * 0.9, 0.1)
                
                # Restart from best point if stuck
                if no_improve >= no_improve_limit:
                    x = best_x + sigma * rng.standard_normal(d)
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    calls += 1
                    if calls >= es_budget:
                        break
                    no_improve = 0
                    sigma = 2.0

        # Phase 2: Nelder-Mead polish from best point
        if time_ok():
            try:
                from scipy.optimize import minimize
                polish_budget = budget - calls
                if polish_budget > 10:
                    res = minimize(
                        lambda v: func(np.clip(v, -5, 5)),
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': polish_budget, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                    )
                    if res.fun < best_fx:
                        best_x = np.clip(res.x, -5, 5)
                        best_fx = res.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
