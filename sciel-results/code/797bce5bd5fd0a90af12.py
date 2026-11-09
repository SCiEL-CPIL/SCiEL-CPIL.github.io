import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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

        # Phase 1: Global exploration with differential evolution (vectorized, uses many calls efficiently)
        # Use a subset of budget for global search
        de_budget = min(200, budget - 150)  # Leave room for local refinement
        if de_budget > 20:
            try:
                bounds = [(-4.0, 4.0)] * dim  # Min guaranteed in [-4,4]^5
                # Use a smaller population to fit within budget
                pop_size = min(20, dim * 2)
                de_result = differential_evolution(
                    safe_func,
                    bounds,
                    maxiter=10,
                    popsize=pop_size,
                    tol=1e-6,
                    seed=42,
                    polish=False,  # We'll do our own local refinement
                    update_interval=1
                )
                if np.isfinite(de_result.fun):
                    best_x = de_result.x.copy()
                    best_f = de_result.fun
            except Exception:
                pass

        if not np.isfinite(best_f):
            # Fallback: simple random sampling
            rng = np.random.default_rng(123)
            for _ in range(min(50, budget - 10)):
                if time.time() > deadline:
                    break
                x = rng.uniform(-4, 4, dim)
                safe_func(x)

        if not np.isfinite(best_f):
            return best_x, best_f

        # Phase 2: Local refinement with L-BFGS-B from multiple starting points
        # Start from best found, plus a few nearby points
        local_starts = [best_x.copy()]
        rng_local = np.random.default_rng(456)
        for _ in range(3):
            x_pert = best_x + rng_local.normal(0, 0.5, dim)
            x_pert = np.clip(x_pert, -5.0, 5.0)
            local_starts.append(x_pert)

        for x0 in local_starts:
            if calls >= budget - 20 or time.time() > deadline:
                break
            try:
                result = minimize(
                    safe_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'maxfun': 50, 'gtol': 1e-8}
                )
                if np.isfinite(result.fun):
                    # safe_func already updates best_x and best_f
                    pass
            except Exception:
                pass

        # Phase 3: Final refinement with Nelder-Mead for fine-tuning
        if calls < budget - 10 and time.time() < deadline:
            try:
                result = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10, 'maxfev': min(50, budget - calls)}
                )
            except Exception:
                pass

        # Phase 4: Final random perturbations around best point
        remaining = budget - calls
        if remaining > 5 and time.time() < deadline:
            rng_final = np.random.default_rng(999)
            for _ in range(min(remaining - 5, 20)):
                if calls >= budget - 2 or time.time() > deadline:
                    break
                scale = 0.05 * (1.0 / (1.0 + 0.1 * _))
                x_pert = best_x + rng_final.normal(0, scale, dim)
                x_pert = np.clip(x_pert, -5.0, 5.0)
                safe_func(x_pert)

        return best_x, best_f
