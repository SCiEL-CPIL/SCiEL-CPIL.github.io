import numpy as np
import scipy.optimize
import scipy.stats
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0  # Conservative: 330s kill, 55s margin
        call_count = 0
        best_f = float('inf')
        best_x = None

        def wrapped_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= self.budget or time.time() > deadline:
                return 1e10
            call_count += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                fx = float(func(x_clipped))
                if np.isnan(fx) or np.isinf(fx):
                    return 1e10
                if fx < best_f:
                    best_f = fx
                    best_x = x_clipped
                return fx
            except Exception:
                return 1e10

        dim = self.dim
        budget = self.budget

        # Phase 1: Sobol initialization - 20 points for good coverage
        n_initial = 20
        try:
            sobol = scipy.stats.qmc.Sobol(d=dim, scramble=True, seed=42)
            points_01 = sobol.random(n_initial)
            points = points_01 * 10.0 - 5.0
        except Exception:
            rng = np.random.default_rng(42)
            points = rng.uniform(-5, 5, (n_initial, dim))

        candidates = []
        for x in points:
            if call_count >= budget:
                break
            fx = wrapped_func(x)
            candidates.append((fx, x))

        if not candidates:
            return best_x, best_f

        # Sort by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES on top 3 candidates (derivative-free, robust)
        top_k = min(3, len(candidates))
        remaining_budget = budget - call_count
        # CMA-ES budget per run: allocate evenly, minimum 30
        cma_budget_per = max(30, remaining_budget // top_k)

        local_results = []
        for i in range(top_k):
            if call_count >= budget or time.time() > deadline:
                break
            start_x = candidates[i][1]
            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    start_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'maxfev': cma_budget_per, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if res.fun < 1e9:
                    local_results.append((res.fun, res.x))
            except Exception:
                local_results.append((candidates[i][0], start_x))

        # Update best
        if local_results:
            local_results.sort(key=lambda t: t[0])
            best_local_f, best_local_x = local_results[0]
            if best_local_f < best_f:
                best_f = best_local_f
                best_x = best_local_x

        # Phase 3: Final polish with Powell on the single best point
        if call_count < budget - 10 and time.time() < deadline - 5:
            try:
                remaining = budget - call_count
                res = scipy.optimize.minimize(
                    wrapped_func,
                    best_x,
                    method='Powell',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 30, 'maxfev': min(80, remaining), 'disp': False}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
