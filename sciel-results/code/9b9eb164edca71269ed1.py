import time
import numpy as np

class Optimizer:
    """
    Multi-start (1+1)-ES with adaptive step size for BBOB Weierstrass (f16).
    Phase 1: Coarse exploration with multiple short (1+1)-ES runs from random starts.
    Phase 2: Fine polishing with a longer (1+1)-ES from the best solution.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0

        n_calls = [0]

        def safe_func(x):
            if n_calls[0] >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            try:
                return float(func(x))
            except Exception:
                return np.inf

        best_x = np.zeros(dim)
        best_f = np.inf

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start coarse exploration
        # 25 restarts x 12 steps = 300 calls for exploration
        n_restarts = 25
        steps_per_restart = 12

        for r in range(n_restarts):
            if n_calls[0] >= budget or time.time() > deadline:
                break

            x = rng.uniform(-5.0, 5.0, size=dim)
            f_x = safe_func(x)

            if not np.isfinite(f_x):
                continue

            if f_x < best_f:
                best_f = f_x
                best_x = x.copy()

            sigma = 1.0
            c_sigma = 1.0 / (5 * dim)
            n_success = 0
            n_total = 0
            window_size = 5 * dim + 1

            for step in range(steps_per_restart):
                if n_calls[0] >= budget or time.time() > deadline:
                    break

                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5.0, 5.0)

                f_trial = safe_func(x_trial)

                if not np.isfinite(f_trial):
                    break

                n_total += 1
                if f_trial < f_x:
                    n_success += 1
                    x = x_trial
                    f_x = f_trial
                    if f_x < best_f:
                        best_f = f_x
                        best_x = x_trial.copy()

                if n_total >= window_size:
                    current_sr = n_success / n_total
                    sigma *= np.exp(c_sigma * (current_sr - 0.5))
                    sigma = np.clip(sigma, 0.01, 5.0)
                    n_success = 0
                    n_total = 0

        # Phase 2: Fine polishing from best solution
        remaining = budget - n_calls[0]
        if remaining > 10 and time.time() < deadline and np.isfinite(best_f):
            x = best_x.copy()
            f_x = best_f
            sigma = 0.1
            c_sigma = 1.0 / (5 * dim)
            n_success = 0
            n_total = 0
            window_size = 5 * dim + 1

            while n_calls[0] < budget and time.time() < deadline:
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5.0, 5.0)

                f_trial = safe_func(x_trial)

                if not np.isfinite(f_trial):
                    break

                n_total += 1
                if f_trial < f_x:
                    n_success += 1
                    x = x_trial
                    f_x = f_trial
                    best_f = f_x
                    best_x = x_trial.copy()

                if n_total >= window_size:
                    current_sr = n_success / n_total
                    sigma *= np.exp(c_sigma * (current_sr - 0.5))
                    sigma = np.clip(sigma, 1e-6, 1.0)
                    n_success = 0
                    n_total = 0

        return best_x, best_f
