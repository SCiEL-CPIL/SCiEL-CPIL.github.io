import numpy as np
import time
from scipy.optimize import minimize, dual_annealing

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                raise TimeoutError()
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if not np.isfinite(fx):
                fx = float('inf')
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global exploration with Dual Annealing
        # Dual Annealing is often more robust than Differential Evolution for smooth,
        # multi-modal landscapes like Rosenbrock, especially in low dimensions.
        de_budget = int(budget * 0.6)  # 60% of budget for global search
        if de_budget > 20:
            try:
                bounds = [(-4.0, 4.0)] * dim  # Min guaranteed in [-4,4]^5
                # maxiter and maxfev control the internal loops
                # We want to use up to de_budget calls
                da_result = dual_annealing(
                    safe_func,
                    bounds,
                    maxiter=100,
                    maxfev=de_budget,
                    seed=42,
                    restarts=5
                )
                if np.isfinite(da_result.fun):
                    # safe_func already updates best_x and best_f
                    # da_result.x might be slightly better or worse depending on the last call
                    # We rely on best_x tracked in safe_func
                    pass
            except Exception:
                pass

        if not np.isfinite(best_f):
            # Fallback: simple random sampling if DA fails completely
            rng = np.random.default_rng(123)
            for _ in range(min(50, budget - 10)):
                if time.time() > deadline:
                    break
                x = rng.uniform(-4, 4, dim)
                safe_func(x)

        if not np.isfinite(best_f):
            return best_x, best_f

        # Phase 2: Local refinement with L-BFGS-B
        # L-BFGS-B is very efficient for the final convergence in the valley.
        local_budget = budget - calls
        if local_budget > 10:
            try:
                result = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 100,
                        'maxfun': local_budget,
                        'gtol': 1e-10,
                        'ftol': 1e-10
                    }
                )
                # safe_func updates best_x and best_f
            except Exception:
                pass

        # Phase 3: Final refinement with Powell (derivative-free, robust)
        # Powell can sometimes squeeze out more precision than L-BFGS-B if gradients are noisy
        # or if L-BFGS-B stops early due to tolerance issues.
        remaining = budget - calls
        if remaining > 5 and time.time() < deadline:
            try:
                result = minimize(
                    safe_func,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': 50,
                        'maxfev': remaining,
                        'xtol': 1e-10,
                        'ftol': 1e-10
                    }
                )
            except Exception:
                pass

        return best_x, best_f
