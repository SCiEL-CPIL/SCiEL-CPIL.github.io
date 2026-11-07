"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize


class Optimizer:
    """Global Differential Evolution followed by local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        bounds = [(-5, 5)] * n

        calls_made = 0

        def evaluate(x):
            nonlocal calls_made
            if calls_made >= budget:
                return 1e10  # Penalize to stop search
            x = np.clip(np.asarray(x, dtype=float).reshape(-1), -5, 5)
            fx = float(func(x))
            calls_made += 1
            return fx

        # Phase 1: Global search with Differential Evolution
        # Use ~60% of budget for global exploration
        # popsize=10, maxiter=30 -> ~300 evaluations (30% of 1000? No, budget is 500)
        # 60% of 500 is 300.
        de_max_iter = max(1, int(budget * 0.6 / 10)) 
        popsize = 10
        
        try:
            de_result = differential_evolution(
                evaluate,
                bounds,
                maxiter=de_max_iter,
                popsize=popsize,
                tol=0.01,
                init='sobol',
                seed=42
            )
            best_x = de_result.x
            best_f = de_result.fun
        except Exception:
            # Fallback if DE fails
            best_x = np.zeros(n)
            best_f = evaluate(best_x)

        # Phase 2: Local refinement with L-BFGS-B
        if calls_made < budget:
            remaining_budget = budget - calls_made
            try:
                local_result = minimize(
                    evaluate,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'maxfun': remaining_budget}
                )
                if local_result.fun < best_f:
                    best_f = local_result.fun
                    best_x = local_result.x
            except Exception:
                pass

        # Phase 3: Final polish if budget remains
        if calls_made < budget:
            try:
                local_result2 = minimize(
                    evaluate,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'maxfev': budget - calls_made}
                )
                if local_result2.fun < best_f:
                    best_f = local_result2.fun
                    best_x = local_result2.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
