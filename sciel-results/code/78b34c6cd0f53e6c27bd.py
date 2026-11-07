import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_f = float('inf')
        best_x = None
        start_time = time.time()
        deadline = start_time + 290.0

        def wrapper(x):
            nonlocal calls_made
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            calls_made += 1
            x = np.clip(x, -5, 5)
            return func(x)

        rng = np.random.default_rng(42)

        # Phase 1: Dense Halton sequence for initial coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            points = np.zeros((n, dim))
            for i in range(n):
                for j in range(dim):
                    points[i, j] = halton(i + 1, j + 2)
            return points

        n_halton = 40
        halton_points = halton_sequence(n_halton, dim)
        halton_points = halton_points * 10 - 5

        for x in halton_points:
            if calls_made >= total_budget or time.time() > deadline:
                break
            try:
                fx = wrapper(x)
            except TimeoutError:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = wrapper(best_x)
            except TimeoutError:
                return np.zeros(dim), float('inf')

        # Phase 2: 1+1-ES with adaptive sigma
        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        success_rate_window = 20
        success_history = []
        target_success_rate = 0.2
        sigma_min = 1e-6
        sigma_max = 5.0

        es_count = 0
        nm_interval = 25
        nm_maxfev = 60

        while calls_made < total_budget and time.time() < deadline:
            # Periodic Nelder-Mead polish
            if es_count > 0 and es_count % nm_interval == 0 and (total_budget - calls_made) > nm_maxfev + 10:
                try:
                    res = minimize(wrapper, best_x, method='Nelder-Mead',
                                   options={
                                       'maxfev': nm_maxfev,
                                       'xatol': 1e-9,
                                       'fatol': 1e-9,
                                       'adaptive': True
                                   })
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                        x = best_x.copy()
                        fx = best_f
                        sigma = 0.3
                        success_history = []
                except TimeoutError:
                    break
                except Exception:
                    pass

            if time.time() > deadline or calls_made >= total_budget:
                break

            # 1+1-ES step
            step = sigma * rng.standard_normal(dim)
            y = x + step
            y = np.clip(y, -5, 5)

            try:
                fy = wrapper(y)
            except TimeoutError:
                break

            if fy < fx:
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                success_history.append(1)
            else:
                success_history.append(0)

            # Adapt sigma based on success rate
            if len(success_history) >= success_rate_window:
                success_rate = sum(success_history[-success_rate_window:]) / success_rate_window
                if success_rate > target_success_rate:
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                sigma = np.clip(sigma, sigma_min, sigma_max)
                # Keep window size constant
                if len(success_history) > success_rate_window * 2:
                    success_history = success_history[-success_rate_window:]

            es_count += 1

            # Periodic coordinate descent polish
            if es_count > 0 and es_count % 30 == 0 and (total_budget - calls_made) > 20:
                for d in range(dim):
                    if time.time() > deadline or calls_made >= total_budget:
                        break
                    # Try a few steps in each direction
                    for direction in [1, -1]:
                        if time.time() > deadline or calls_made >= total_budget:
                            break
                        h = 0.01
                        x_trial = x.copy()
                        x_trial[d] += direction * h
                        x_trial = np.clip(x_trial, -5, 5)
                        try:
                            f_trial = wrapper(x_trial)
                        except TimeoutError:
                            break
                        if f_trial < fx:
                            x = x_trial
                            fx = f_trial
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                        else:
                            break

        # Phase 3: Final high-precision polish
        if calls_made < total_budget - 10 and time.time() < deadline - 5:
            try:
                res = minimize(wrapper, best_x, method='Nelder-Mead',
                               options={
                                   'maxfev': min(80, total_budget - calls_made - 5),
                                   'xatol': 1e-10,
                                   'fatol': 1e-10,
                                   'adaptive': True
                               })
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except TimeoutError:
                pass
            except Exception:
                pass

        return best_x, best_f
