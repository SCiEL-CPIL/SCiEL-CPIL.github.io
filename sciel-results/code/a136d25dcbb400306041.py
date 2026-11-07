import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        max_calls = self.budget
        calls_made = 0
        dim = self.dim

        # Wrapper for counting and clipping
        def eval_func(x):
            nonlocal calls_made
            if calls_made >= max_calls:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_made += 1
            return val

        # (1+1)-ES with aggressive sigma adaptation
        rng = np.random.default_rng(12345)
        x = np.zeros(dim)
        fx = eval_func(x)

        sigma = 2.0  # larger initial step to explore the valley
        success_hist = []
        hist_len = 15

        # Use 60% of budget for ES, 40% for L-BFGS-B polish
        es_budget = int(self.budget * 0.6)

        for i in range(es_budget - 1):
            if time.time() > deadline:
                break
            if calls_made >= max_calls:
                break

            y = x + sigma * rng.standard_normal(dim)
            fy = eval_func(y)

            if fy < fx:
                x, fx = y, fy
                success_hist.append(1)
            else:
                success_hist.append(0)

            if len(success_hist) > hist_len:
                success_hist.pop(0)

            # Adapt sigma: aim for ~30% success rate
            if len(success_hist) == hist_len:
                success_rate = np.mean(success_hist)
                if success_rate > 0.3:
                    sigma *= 1.15
                elif success_rate < 0.15:
                    sigma *= 0.85
                else:
                    sigma *= 1.0

        # L-BFGS-B polish
        if calls_made < max_calls and time.time() < deadline:
            nm_calls = 0
            nm_limit = max_calls - calls_made

            def lbfgs_func(z):
                nonlocal nm_calls
                if nm_calls >= nm_limit:
                    return np.inf
                nm_calls += 1
                return func(np.clip(z, -5.0, 5.0))

            try:
                res = minimize(lbfgs_func, x, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0)] * dim,
                               options={'maxiter': max(1, nm_limit // 2),
                                        'maxfun': nm_limit,
                                        'ftol': 1e-12,
                                        'gtol': 1e-8})
                if res.fun < fx:
                    x, fx = res.x, res.fun
            except Exception:
                pass

        return x, fx
