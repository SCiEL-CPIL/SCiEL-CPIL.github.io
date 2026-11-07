"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start 1+1 ES with adaptive step size for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        total_calls = 0

        def try_eval(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return None
            if time.time() > deadline - 3:
                return None
            x = np.clip(x, -5, 5)
            f = func(x)
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def one_plus_one_es(x0, sigma0, max_calls):
            """1+1 ES with adaptive step size."""
            nonlocal total_calls, best_x, best_f
            if max_calls < 5:
                return x0, best_f

            x = x0.copy()
            sigma = sigma0
            f_curr = best_f
            # We don't know f_curr exactly, so evaluate x0 first
            f_x0 = try_eval(x)
            if f_x0 is None:
                return x, best_f
            f_curr = f_x0

            success_rate = 0.5
            for _ in range(max_calls):
                if total_calls >= budget or time.time() > deadline - 5:
                    break

                # Generate one mutant
                z = rng.standard_normal(n)
                x_trial = x + sigma * z
                f_trial = try_eval(x_trial)
                if f_trial is None:
                    break

                if f_trial < f_curr:
                    # Success
                    x = x_trial
                    f_curr = f_trial
                    success_rate = 0.7 * success_rate + 0.3
                    # Increase sigma slightly
                    sigma *= 1.1
                else:
                    # Failure
                    success_rate = 0.7 * success_rate
                    # Decrease sigma
                    sigma *= 0.9

                # Adaptive sigma adjustment based on success rate
                if success_rate > 0.65:
                    sigma *= 1.2
                elif success_rate < 0.35:
                    sigma *= 0.8

                # Keep sigma in reasonable bounds
                sigma = max(sigma, 1e-6)
                sigma = min(sigma, 10.0)

            return x, best_f

        # --- Phase 1: Initial broad sampling ---
        n_initial = min(40, budget // 5)
        # Structured points
        structured_points = [
            np.zeros(n),
            np.ones(n) * 2,
            -np.ones(n) * 2,
            np.ones(n) * 1,
            -np.ones(n) * 1,
            np.array([3.0, -3.0, 2.0, -2.0, 1.0]),
            np.array([-2.0, 3.0, -1.0, 2.0, -3.0]),
            np.array([1.5, 1.5, 1.5, 1.5, 1.5]),
            np.array([-1.5, -1.5, -1.5, -1.5, -1.5]),
            np.array([4.0, -4.0, 3.0, -3.0, 2.0]),
            np.array([-3.0, 4.0, -2.0, 3.0, -1.0]),
        ]
        for p in structured_points:
            if total_calls >= budget:
                break
            try_eval(p)

        # Random sampling with decreasing scale
        for i in range(n_initial):
            if total_calls >= budget or time.time() > deadline - 5:
                break
            scale = 4.0 * (1.0 - i / max(1, n_initial)) + 0.5
            x_try = rng.normal(0, scale, n)
            try_eval(x_try)

        if best_x is None:
            return None, np.inf

        # --- Phase 2: Multi-start 1+1 ES ---
        # Collect diverse starting points
        starts = [best_x.copy()]
        
        # Add more diverse starts from random sampling
        n_diverse = min(8, (budget - total_calls) // 20)
        for _ in range(n_diverse):
            if total_calls >= budget - 10:
                break
            x_rand = rng.uniform(-4, 4, n)
            f_rand = try_eval(x_rand)
            if f_rand is not None:
                starts.append(x_rand.copy())

        # Sort starts by function value (best first)
        starts.sort(key=lambda x: best_f if np.allclose(x, best_x) else np.inf)

        # Run 1+1 ES on each start
        remaining_budget = budget - total_calls
        budget_per_start = max(15, remaining_budget // max(1, len(starts)))
        
        for i, x_start in enumerate(starts):
            if total_calls >= budget - 5 or time.time() > deadline - 10:
                break
            remaining = budget - total_calls
            alloc = min(budget_per_start, remaining)
            if alloc < 10:
                break
            sigma0 = 0.5 if i == 0 else 0.3
            x_opt, f_opt = one_plus_one_es(x_start, sigma0, alloc)
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # --- Phase 3: Final polish ---
        if total_calls < budget and best_x is not None and time.time() < deadline - 5:
            # Run 1+1 ES from best point with small sigma
            if budget - total_calls >= 20:
                x_opt, f_opt = one_plus_one_es(best_x.copy(), 0.05, min(40, budget - total_calls))
                if f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt.copy()

            # Tiny perturbations
            n_final = min(10, budget - total_calls)
            for i in range(n_final):
                if total_calls >= budget or time.time() > deadline - 3:
                    break
                perturbation = rng.normal(0, 0.01, n)
                x_try = best_x + perturbation
                f_try = try_eval(x_try)

        return best_x, best_f
# EVOLVE-BLOCK-END
