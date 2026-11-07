import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart (1+1)-ES with Nelder-Mead refinement for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        used_calls = 0

        def check_time():
            return time.time() > deadline

        def try_point(x):
            nonlocal best_x, best_f, used_calls
            if used_calls >= budget or check_time():
                return
            fx = func(x)
            used_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Initial exploration
        try_point(np.zeros(d))

        # Phase 2: Multi-restart (1+1)-ES with adaptive step size
        num_restarts = 6
        calls_per_restart = max(15, (budget - used_calls) // num_restarts)
        sigma = 2.0
        sigma_min = 1e-6
        sigma_max = 5.0
        alpha = 0.85
        beta = 1.5

        for restart in range(num_restarts):
            if used_calls >= budget or check_time():
                break

            if restart == 0 and best_x is not None:
                x = best_x.copy()
            else:
                x = rng.uniform(-4, 4, d)
                try_point(x)

            for _ in range(calls_per_restart):
                if used_calls >= budget or check_time():
                    break

                y = x + sigma * rng.standard_normal(d)
                y = np.clip(y, -5, 5)

                fy = func(y)
                used_calls += 1

                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

                if fy < best_f + 1e-12:
                    x = y.copy()
                    sigma = min(sigma * 1.1, sigma_max)
                else:
                    sigma = max(sigma * alpha, sigma_min)

        # Phase 3: Nelder-Mead refinement from best point
        if best_x is not None and used_calls < budget and not check_time():
            calls_made = [0]
            max_local_calls = budget - used_calls

            def wrapped_func(x):
                nonlocal best_f, best_x
                calls_made[0] += 1
                if calls_made[0] > max_local_calls or check_time():
                    return 1e10
                fx = func(x)
                used_calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx

            try:
                res = minimize(wrapped_func, best_x.copy(), method='Nelder-Mead',
                               options={'xatol': 1e-8, 'fatol': 1e-10,
                                        'maxiter': max_local_calls,
                                        'adaptive': True})
            except Exception:
                pass

        # Phase 4: Final fine-tuning with very small steps
        if best_x is not None and used_calls < budget and not check_time():
            x = best_x.copy()
            fx = best_f
            remaining = budget - used_calls

            for scale in [0.05, 0.01, 0.001, 0.0001]:
                if used_calls >= budget or check_time():
                    break
                for _ in range(min(40, remaining)):
                    if used_calls >= budget or check_time():
                        break
                    z = rng.standard_normal(d)
                    y = np.clip(x + scale * z, -5, 5)
                    try_point(y)
                    if best_x is not None and best_f < fx:
                        x = best_x.copy()
                        fx = best_f

        return best_x, best_f
