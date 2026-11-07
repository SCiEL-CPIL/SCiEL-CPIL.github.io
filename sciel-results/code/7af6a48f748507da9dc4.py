import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        max_calls = self.budget
        calls_made = 0
        dim = self.dim

        # Track best solution
        best_x = np.zeros(dim)
        best_fx = np.inf

        # Wrapper for counting calls
        def eval_func(x):
            nonlocal calls_made, best_x, best_fx
            if calls_made >= max_calls:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_made += 1
            if val < best_fx:
                best_fx = val
                best_x = x_clipped.copy()
            return val

        # Multi-restart (1+1)-ES with CMA-like adaptation
        rng = np.random.default_rng(42)

        # Phase 1: Global exploration with multiple restarts
        # Use 70% of budget for exploration, 30% for local polish
        explore_budget = int(self.budget * 0.70)
        polish_budget = max_calls - explore_budget

        # Number of restarts based on budget
        n_restarts = min(8, max(2, explore_budget // 40))
        budget_per_restart = explore_budget // n_restarts

        for r in range(n_restarts):
            if calls_made >= max_calls or time.time() > deadline:
                break

            # Random initialization within the box
            if r == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4.0, 4.0, size=dim)

            fx = eval_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            # (1+1)-ES with sigma adaptation for this restart
            sigma = 1.5
            hist_len = 10
            success_hist = []
            iters_this_restart = 0
            max_iters_restart = budget_per_restart - 1

            for i in range(max_iters_restart):
                if time.time() > deadline:
                    break
                if calls_made >= max_calls:
                    break
                if len(success_hist) >= hist_len:
                    # Check if stuck (no improvement in recent history)
                    if np.mean(success_hist) < 0.05:
                        break

                # Generate offspring
                y = x + sigma * rng.standard_normal(dim)
                fy = eval_func(y)

                if fy < fx:
                    x, fx = y, fy
                    success_hist.append(1)
                    # Update global best
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    success_hist.append(0)

                if len(success_hist) > hist_len:
                    success_hist.pop(0)

                # Adapt sigma: target ~25% success rate
                if len(success_hist) == hist_len:
                    sr = np.mean(success_hist)
                    if sr > 0.35:
                        sigma *= 1.2
                    elif sr < 0.15:
                        sigma *= 0.8
                    else:
                        sigma *= 1.0

                # Clamp sigma
                sigma = np.clip(sigma, 0.01, 5.0)

                iters_this_restart += 1

        # Phase 2: Local polish with L-BFGS-B from best point
        if calls_made < max_calls and time.time() < deadline:
            nm_calls = 0
            nm_limit = max_calls - calls_made

            def lbfgs_func(z):
                nonlocal nm_calls, best_x, best_fx
                if nm_calls >= nm_limit:
                    return np.inf
                nm_calls += 1
                z_clipped = np.clip(z, -5.0, 5.0)
                val = func(z_clipped)
                if val < best_fx:
                    best_fx = val
                    best_x = z_clipped.copy()
                return val

            try:
                res = minimize(lbfgs_func, best_x, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0)] * dim,
                               options={'maxiter': max(1, nm_limit // 3),
                                        'maxfun': nm_limit,
                                        'ftol': 1e-14,
                                        'gtol': 1e-10,
                                        'maxls': 50})
            except Exception:
                pass

        # Phase 3: If we have time and calls left, do a few more L-BFGS-B from different starts
        if calls_made < max_calls and time.time() < deadline:
            # Try a few more local optimizations from random points
            n_extra = min(3, (max_calls - calls_made) // 20)
            for k in range(n_extra):
                if calls_made >= max_calls or time.time() > deadline:
                    break
                x0 = rng.uniform(-4.0, 4.0, size=dim)
                nm_calls2 = 0
                nm_limit2 = max_calls - calls_made

                def lbfgs_func2(z, x0=x0, nm_calls2=nm_calls2):
                    nonlocal nm_calls2, best_x, best_fx
                    if nm_calls2 >= nm_limit2:
                        return np.inf
                    nm_calls2 += 1
                    z_clipped = np.clip(z, -5.0, 5.0)
                    val = func(z_clipped)
                    if val < best_fx:
                        best_fx = val
                        best_x = z_clipped.copy()
                    return val

                try:
                    res = minimize(lbfgs_func2, x0, method='L-BFGS-B',
                                   bounds=[(-5.0, 5.0)] * dim,
                                   options={'maxiter': max(1, nm_limit2 // 5),
                                            'maxfun': nm_limit2,
                                            'ftol': 1e-14,
                                            'gtol': 1e-10})
                except Exception:
                    pass

        return best_x, best_fx
