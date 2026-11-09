"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Random restart + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        calls = 0
        
        # Phase 1: Random Search to find a good starting point
        # Use 10% of budget for this
        n_random = max(10, budget // 10)
        
        best_x = None
        best_f = np.inf
        
        for _ in range(n_random):
            if time.time() > deadline:
                break
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # If no calls made (shouldn't happen), fallback to 0
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        # Phase 2: Nelder-Mead Polish
        # Use remaining budget
        remaining = budget - calls
        
        if remaining > 10:
            try:
                from scipy.optimize import minimize
                
                # Estimate maxiter based on remaining budget
                # NM uses ~ dim+1 calls per iteration
                max_iter = max(10, remaining // (dim + 1))
                
                res = minimize(
                    lambda z: func(np.clip(z, -5, 5)),
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    },
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
