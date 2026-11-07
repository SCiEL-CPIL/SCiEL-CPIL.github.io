"""Multi-start L-BFGS-B with high-precision polish for BBOB f12 (bent cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start L-BFGS-B with final polish for BBOB f12 (bent cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track global best
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        def check_budget():
            nonlocal total_calls
            if total_calls >= budget or time.time() > deadline:
                return False
            return True

        from scipy.optimize import minimize

        # First, evaluate center point
        center = np.zeros(dim)
        f_center = func(center)
        total_calls += 1
        best_f = f_center
        best_x = center.copy()

        # Reserve ~30 calls for final polish
        polish_budget = 30
        search_budget = budget - 1 - polish_budget
        n_restarts = min(12, max(1, search_budget // 35))
        evals_per_restart = max(20, search_budget // n_restarts)

        for r in range(n_restarts):
            if not check_budget():
                break
            if time.time() > deadline - 10.0:  # Reserve time for polish
                break
            
            # Random starting point, biased toward center for bent cigar
            x0 = rng.uniform(-4.0, 4.0, dim)
            
            def objective(x):
                nonlocal total_calls, best_x, best_f
                if total_calls >= budget or time.time() > deadline:
                    return best_f
                total_calls += 1
                val = func(x)
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val

            try:
                res = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 100,
                        'maxfun': evals_per_restart,
                        'gtol': 1e-6,
                        'ftol': 1e-12
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # High-precision polish phase from the best point found
        if check_budget() and time.time() < deadline - 5.0:
            def polish_objective(x):
                nonlocal total_calls, best_x, best_f
                if total_calls >= budget or time.time() > deadline:
                    return best_f
                total_calls += 1
                val = func(x)
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val

            try:
                res_polish = minimize(
                    polish_objective,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 200,
                        'maxfun': polish_budget,
                        'gtol': 1e-10,
                        'ftol': 1e-15
                    }
                )
                if res_polish.fun < best_f:
                    best_f = res_polish.fun
                    best_x = res_polish.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
