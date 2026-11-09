"""Combined: (1+1)-ES with window adaptation + periodic exploration + scipy local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """(1+1)-ES with adaptive step size, periodic exploration, and scipy local polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget

        calls = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= budget:
                raise StopIteration("Budget exceeded")
            x_arr = np.asarray(x, dtype=float)
            f_val = func(x_arr)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x_arr.copy()
            return f_val

        # Initial evaluation
        x_best = rng.uniform(-5, 5, n)
        try:
            fx_best = safe_func(x_best)
        except (TimeoutError, StopIteration):
            return best_x, best_f

        # Parameters
        sigma = 2.0
        sigma_min = 1e-5
        sigma_max = 8.0
        success_ratio = 0.2
        stall_limit = 30
        stall_count = 0
        window_size = 10
        successes_in_window = []

        # Periodic exploration
        explore_interval = 40
        calls_since_explore = 0

        remaining = budget - calls

        # Phase 1: Main (1+1)-ES with periodic exploration
        while remaining > 0 and time.time() < deadline:
            calls_since_explore += 1
            if calls_since_explore >= explore_interval:
                calls_since_explore = 0
                try:
                    x_jump = rng.uniform(-5, 5, n)
                    fx_jump = safe_func(x_jump)
                except (TimeoutError, StopIteration):
                    break
                remaining = budget - calls
                if fx_jump < fx_best:
                    x_best, fx_best = x_jump, fx_jump
                    sigma = min(sigma * 1.5, sigma_max)
                    stall_count = 0
                continue

            # Generate candidate
            y = np.clip(x_best + sigma * rng.standard_normal(n), -5, 5)
            try:
                fy = safe_func(y)
            except (TimeoutError, StopIteration):
                break
            remaining = budget - calls

            if fy < fx_best:
                x_best, fx_best = y, fy
                successes_in_window.append(1)
                stall_count = 0
            else:
                successes_in_window.append(0)

            if len(successes_in_window) > window_size:
                successes_in_window.pop(0)

            if len(successes_in_window) >= window_size:
                success_rate = np.mean(successes_in_window)
                if success_rate > success_ratio:
                    sigma = min(sigma * 1.15, sigma_max)
                elif success_rate < success_ratio:
                    sigma = max(sigma * 0.85, sigma_min)

            if fy >= fx_best:
                stall_count += 1
                if stall_count >= stall_limit:
                    stall_count = 0
                    sigma = 2.0
                    try:
                        x_new = rng.uniform(-5, 5, n)
                        fx_new = safe_func(x_new)
                    except (TimeoutError, StopIteration):
                        break
                    remaining = budget - calls
                    if fx_new < fx_best:
                        x_best, fx_best = x_new, fx_new
                    successes_in_window = []

        # Phase 2: Local refinement with scipy
        if best_x is not None and calls < budget and time.time() < deadline:
            remaining_calls = budget - calls

            # L-BFGS-B for smooth local refinement
            if remaining_calls > 20:
                max_iter = max(10, remaining_calls // 2)
                try:
                    scipy.optimize.minimize(
                        safe_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * n,
                        options={
                            'maxiter': max_iter,
                            'ftol': 1e-12,
                            'gtol': 1e-10,
                            'maxls': 30,
                        },
                    )
                except (TimeoutError, StopIteration, Exception):
                    pass

            # Nelder-Mead polish with tight tolerance
            if calls < budget and time.time() < deadline:
                remaining_calls = budget - calls
                if remaining_calls > 10:
                    max_iter = max(5, remaining_calls // (n + 2))
                    try:
                        scipy.optimize.minimize(
                            safe_func,
                            best_x,
                            method='Nelder-Mead',
                            options={
                                'maxiter': max_iter,
                                'xatol': 1e-8,
                                'fatol': 1e-10,
                                'adaptive': True,
                            },
                        )
                    except (TimeoutError, StopIteration, Exception):
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END
