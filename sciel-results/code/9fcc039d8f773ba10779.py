"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """Hybrid Differential Evolution and Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self.calls_made = 0

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self.calls_made = 0
        
        def counted_func(x):
            if self.calls_made >= budget:
                return float('inf')
            self.calls_made += 1
            return func(np.clip(x, -5, 5))

        best_x = None
        best_f = float('inf')

        # Phase 1: Global Search with Differential Evolution
        # Allocate ~60% of budget to DE
        de_budget = int(budget * 0.6)
        # popsize=10 in 5D means ~50 evals per iteration
        maxiter_de = max(1, de_budget // 50)
        
        try:
            res_de = differential_evolution(
                counted_func,
                bounds=[(-5, 5)] * dim,
                maxiter=maxiter_de,
                popsize=10,
                seed=42,
                tol=1e-8
            )
            if res_de.fun < best_f:
                best_f = res_de.fun
                best_x = res_de.x.copy()
        except Exception:
            pass

        # Phase 2: Local Refinement with Nelder-Mead
        remaining_budget = budget - self.calls_made
        
        if remaining_budget > 10 and best_x is not None:
            # Estimate maxiter for NM
            # NM in 5D uses ~5-10 calls per iter
            maxiter_nm = remaining_budget // 10
            
            try:
                res_nm = minimize(
                    counted_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': maxiter_nm,
                        'xatol': 1e-8,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x.copy()
            except Exception:
                pass
        elif best_x is None:
            # Fallback if DE failed or used all budget
            # Try a quick random search + NM
            rng = np.random.default_rng(42)
            for _ in range(10):
                x = rng.uniform(-5, 5, dim)
                fx = counted_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            if best_x is not None:
                remaining_budget = budget - self.calls_made
                if remaining_budget > 10:
                    maxiter_nm = remaining_budget // 10
                    try:
                        res_nm = minimize(
                            counted_func,
                            best_x,
                            method='Nelder-Mead',
                            options={
                                'maxiter': maxiter_nm,
                                'xatol': 1e-8,
                                'fatol': 1e-12,
                                'adaptive': True
                            }
                        )
                        if res_nm.fun < best_f:
                            best_f = res_nm.fun
                            best_x = res_nm.x.copy()
                    except Exception:
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END
