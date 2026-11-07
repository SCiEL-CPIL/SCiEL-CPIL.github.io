"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-stage optimizer: random sampling, Nelder-Mead local refinement,
    and a final (1+1)-ES polishing phase with adaptive step size.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        n = self.dim
        budget = self.budget

        if budget < 10:
            rng = np.random.default_rng(0)
            best_x = rng.uniform(-5, 5, n)
            best_f = func(best_x)
            for _ in range(budget - 1):
                x = rng.uniform(-5, 5, n)
                f = func(x)
                if f < best_f:
                    best_f, best_x = f, x
            return best_x, best_f

        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            if time.time() > deadline:
                return None
            calls_made += 1
            return func(x)

        # Phase 1: Random sampling - use ~25% of budget
        n_random = max(10, int(budget * 0.25))
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        for _ in range(n_random):
            if calls_made >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            f = safe_call(x)
            if f is None:
                break
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = safe_call(best_x)
            if best_f is None:
                return best_x, np.inf

        # Phase 2: Nelder-Mead from the best random point
        nm_budget = max(50, int(budget * 0.35))
        nm_start = time.time()
        nm_deadline = min(deadline, nm_start + 120.0)

        nm_calls = 0

        def nm_func(x):
            nonlocal nm_calls
            if time.time() > nm_deadline:
                return 1e300
            nm_calls += 1
            if calls_made >= budget:
                return 1e300
            calls_made += 1
            return func(x)

        try:
            from scipy.optimize import minimize

            x0 = best_x
            initial_simplex = [x0]
            for i in range(n):
                xi = x0.copy()
                step = 0.5 if x0[i] < 0 else -0.5
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                initial_simplex.append(xi)

            res = minimize(
                nm_func,
                x0,
                method='Nelder-Mead',
                options={
                    'initial_simplex': initial_simplex,
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                }
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

        # Phase 3: (1+1)-ES for final polishing with remaining budget
        remaining = budget - calls_made
        if remaining > 5 and best_x is not None:
            sigma = 0.5
            x = best_x.copy()
            fx = best_f
            fail_count = 0

            for _ in range(remaining):
                if calls_made >= budget or time.time() > deadline:
                    break

                z = rng.standard_normal(n)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5, 5)

                f_trial = safe_call(x_trial)
                if f_trial is None:
                    break

                if f_trial < fx:
                    x = x_trial
                    fx = f_trial
                    fail_count = 0
                    sigma *= 1.05
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    fail_count += 1
                    sigma *= 0.95

                sigma = np.clip(sigma, 1e-6, 5.0)

                if fail_count > 20:
                    sigma = 1.0
                    fail_count = 0
                    x_trial = x + 0.5 * rng.standard_normal(n)
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = safe_call(x_trial)
                    if f_trial is not None and f_trial < fx:
                        x = x_trial
                        fx = f_trial
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
