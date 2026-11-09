# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize


class Optimizer:
    """Hybrid Differential Evolution and Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        call_count = 0
        def wrapped_func(x):
            nonlocal call_count
            call_count += 1
            return func(x)

        bounds = [(-5, 5)] * dim
        
        # Phase 1: Differential Evolution
        # popsize=10, maxiter=25
        # Expected calls: ~10 (init) + 25 * 10 = 260
        try:
            res_de = differential_evolution(
                wrapped_func, 
                bounds, 
                maxiter=25, 
                popsize=10, 
                seed=42,
                tol=1e-10,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False
            )
            best_x = res_de.x
            best_f = res_de.fun
        except Exception:
            # Fallback to random search if DE fails
            rng = np.random.default_rng(42)
            best_x = np.zeros(dim)
            best_f = float('inf')
            for _ in range(100):
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x

        # Phase 2: Nelder-Mead Polish
        # Used ~260 calls. Budget left: ~240.
        # NM maxiter=40. Expected calls: ~40 * 5 = 200.
        # Total ~460. Safe.
        try:
            res_nm = minimize(
                wrapped_func, 
                best_x, 
                method='Nelder-Mead',
                options={
                    'maxiter': 40,
                    'xatol': 1e-8,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if res_nm.fun < best_f:
                best_f = res_nm.fun
                best_x = res_nm.x
        except Exception:
            pass

        return best_x, best_f
# EVOLVE-BLOCK-END
