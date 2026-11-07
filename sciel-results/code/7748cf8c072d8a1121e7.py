"""Optimizer using a (1+1)-ES with 1/5-rule and multi-start for BBOB f18."""
import numpy as np
import time

class Optimizer:
    """Multi-start (1+1)-ES with adaptive step-size for ill-conditioned functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _clip(self, x):
        return np.clip(x, -5.0, 5.0)

    def _one_plus_one(self, func, x0, sigma, budget, rng, deadline):
        """Run (1+1)-ES with 1/5-rule step-size adaptation."""
        dim = self.dim
        x = x0.copy()
        fx = func(x)
        calls = 1
        sigma = float(sigma)

        # Track consecutive failures for 1/5 rule
        successes = 0
        total = 0
        # Window for 1/5 rule
        window = 1 + dim  # standard window size

        while calls < budget:
            if time.time() > deadline - 3.0:
                break

            # Generate mutant
            z = rng.standard_normal(dim)
            x_trial = self._clip(x + sigma * z)
            fx_trial = func(x_trial)
            calls += 1

            total += 1
            if fx_trial < fx:
                x = x_trial
                fx = fx_trial
                successes += 1

            # Adapt sigma every 'window' steps
            if total >= window:
                rate = successes / total
                if rate > 0.2:
                    sigma *= 1.25
                elif rate < 0.2:
                    sigma /= 1.25
                sigma = np.clip(sigma, 1e-10, 20.0)
                successes = 0
                total = 0

            # Early termination if sigma is too small
            if sigma < 1e-10:
                break

            # Also check if we're making very slow progress
            # (optional: could add more sophisticated stagnation detection)

        return x, fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Phase 1: Multi-start to find a good basin
        # Use several starting points with different sigmas
        n_starts = 8
        budget_per_start = max(30, budget // (n_starts + 2))

        # Initial sigma for exploration
        sigma_explore = 2.0

        # Generate diverse starting points
        starts = [np.zeros(dim)]
        for i in range(n_starts - 1):
            if i % 2 == 0:
                # Random in box
                x_s = rng.uniform(-4.0, 4.0, dim)
            else:
                # Random with smaller magnitude
                x_s = rng.uniform(-2.0, 2.0, dim)
            starts.append(x_s)

        for i, x0 in enumerate(starts):
            if total_calls >= budget - 10:
                break
            if time.time() > deadline - 10.0:
                break

            remaining = budget - total_calls
            alloc = min(budget_per_start, remaining)

            # Use smaller sigma for later starts (refinement)
            sigma = sigma_explore * (0.8 ** i)

            x_res, f_res, calls_used = self._one_plus_one(
                func, x0, sigma, alloc, rng, deadline
            )
            total_calls += calls_used

            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        # Phase 2: Refinement from the best point found
        if total_calls < budget - 20 and time.time() < deadline - 15.0:
            remaining = budget - total_calls
            # Use a smaller sigma for refinement
            sigma_refine = 0.5
            x_res, f_res, calls_used = self._one_plus_one(
                func, best_x, sigma_refine, remaining, rng, deadline
            )
            total_calls += calls_used

            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        # Phase 3: If still have budget, do a few more local refinements
        if total_calls < budget - 10 and time.time() < deadline - 10.0:
            remaining = budget - total_calls
            # Try a few more perturbations from best
            sigma_final = 0.2
            x_res, f_res, calls_used = self._one_plus_one(
                func, best_x, sigma_final, remaining, rng, deadline
            )
            total_calls += calls_used

            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            total_calls += 1

        return best_x, best_f
