"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """Hybrid CMA-ES (via scipy) with multi-start and local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track best solution and function value
        best_x = None
        best_f = np.inf

        # We'll use a portion of the budget for multi-start CMA-ES-like searches
        # and the rest for local refinement using scipy's L-BFGS-B or Nelder-Mead.

        # Strategy:
        # 1. Do several short runs of a simple (1+lambda)-ES or use scipy's differential_evolution
        #    with limited iterations to find good starting points.
        # 2. Then refine the best point using a local optimizer.

        # Given the budget is 500 and dim=5, we can afford:
        # - A few global searches to find promising regions
        # - Local refinement to polish the solution

        # Let's use a hybrid approach:
        # Phase 1: Multi-start with small populations (like a mini differential evolution)
        # Phase 2: Local refinement using Nelder-Mead or L-BFGS-B

        # Since we can't easily integrate CMA-ES without external libs, let's use
        # scipy's differential_evolution for global search with limited iterations,
        # then refine with a local method.

        # However, differential_evolution might use too many function evaluations.
        # Let's do a manual multi-start approach instead.

        # Phase 1: Multi-start (1+1)-ES with random restarts
        # Use 60% of budget for global exploration
        global_budget = int(budget * 0.6)
        local_budget = budget - global_budget

        # Divide global budget among several restarts
        n_restarts = 5
        budget_per_restart = global_budget // n_restarts

        # Keep track of all good solutions found
        candidates = []

        for r in range(n_restarts):
            # Start from a random point
            x = rng.uniform(-4, 4, dim)  # Minimizer is in [-4,4]^5
            fx = func(x)
            candidates.append((x.copy(), fx))

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.0
            # Run (1+1)-ES for this restart
            for _ in range(budget_per_restart - 1):
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    if sigma < 0.01:
                        break

            candidates.append((x.copy(), fx))

        # Phase 2: Local refinement using Nelder-Mead on the best candidate
        # Use the remaining budget for local optimization
        if best_x is not None:
            # Wrap func to count evaluations and respect budget
            evals_used = 0
            max_local_evals = local_budget

            def local_func(x):
                nonlocal evals_used
                if evals_used >= max_local_evals:
                    return np.inf
                evals_used += 1
                return func(np.clip(x, -5, 5))

            try:
                res = minimize(
                    local_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_local_evals,
                        'maxfev': max_local_evals,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
