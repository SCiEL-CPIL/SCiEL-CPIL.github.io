import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 285.0  # safety margin before 330s kill

        n_calls = [0]
        stop_flag = [False]

        def safe_func(x):
            if stop_flag[0]:
                return 1e12
            n_calls[0] += 1
            if n_calls[0] > budget:
                stop_flag[0] = True
                return 1e12
            if time.time() > deadline:
                stop_flag[0] = True
                return 1e12
            try:
                val = func(np.asarray(x, dtype=np.float64))
                if not np.isfinite(val):
                    return 1e12
                return float(val)
            except Exception:
                return 1e12

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start CMA-ES for global exploration
        # Use ~60% of budget for exploration, 40% for local refinement
        explore_budget = int(budget * 0.55)
        local_budget = budget - explore_budget

        # Generate diverse starting points for multi-start
        n_starts = 6
        starts = []
        # One at origin
        starts.append(np.zeros(dim))
        # One at random center
        starts.append(rng.uniform(-3, 3, dim))
        # Several random points
        for _ in range(n_starts - 2):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        best_x = None
        best_f = 1e18

        # Run CMA-ES from multiple starts, sharing budget
        calls_per_start = max(5, explore_budget // n_starts)

        for s in starts:
            if stop_flag[0] or n_calls[0] >= explore_budget:
                break
            # CMA-ES with small population for 5D
            popsize = min(8, dim + 2)
            maxiter = max(2, calls_per_start // popsize)
            try:
                res = minimize(
                    safe_func,
                    s,
                    method='Nelder-Mead',
                    options={
                        'maxiter': maxiter * 2,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if stop_flag[0]:
                    break
                if res.success or res.fun < best_f:
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
            except Exception:
                pass

        # Phase 2: Multistart Nelder-Mead local refinement
        # Use remaining budget for precise local optimization
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        # Create local starting points around the best found
        local_starts = [best_x.copy()]
        # Add small perturbations for multi-start local search
        for _ in range(4):
            perturbed = best_x + rng.normal(0, 0.5, dim)
            perturbed = np.clip(perturbed, -5, 5)
            local_starts.append(perturbed)

        local_calls_per_start = max(3, local_budget // len(local_starts))

        for ls in local_starts:
            if stop_flag[0] or n_calls[0] >= budget - 2:
                break
            try:
                res = minimize(
                    safe_func,
                    ls,
                    method='Nelder-Mead',
                    options={
                        'maxiter': local_calls_per_start,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if stop_flag[0]:
                    break
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Final check: if we still have budget, do one more pass
        if n_calls[0] < budget - 1 and not stop_flag[0]:
            try:
                res = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': budget - n_calls[0] - 1,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
