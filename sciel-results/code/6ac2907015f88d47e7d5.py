"""Bootstrap portfolio optimizer for BBOB f23 (Katsuura) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-strategy optimizer for Katsuura:
    1) Proper Latin Hypercube sampling for fast initial basin discovery.
    2) (1+1)-ES with adaptive sigma and smart restarts.
    3) Deadline guard to stay within wall-clock limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._deadline_margin = 30.0  # seconds to spare before hard limit

    @staticmethod
    def _latin_hypercube(n, dim, rng):
        """Generate n points in [-5,5]^dim using Latin Hypercube sampling."""
        points = np.zeros((n, dim))
        for d in range(dim):
            # Create n equally spaced intervals and sample one point from each
            intervals = np.linspace(-5, 5, n + 1)
            lower = intervals[:-1]
            upper = intervals[1:]
            # Randomly permute the intervals
            perm = rng.permutation(n)
            # Sample uniformly within each interval
            points[:, d] = lower[perm] + rng.uniform(0, 1, n) * (upper[perm] - lower[perm])
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Proper Latin Hypercube sampling for fast initial discovery
        n_lhs = min(30, budget // 10)
        lhs_points = self._latin_hypercube(n_lhs, dim, rng)
        best_x = None
        best_f = np.inf

        for i in range(n_lhs):
            x = lhs_points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES with adaptive sigma and restarts
        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        no_improve_count = 0
        calls_made = n_lhs
        stagnation_limit = 25
        restart_count = 0
        max_restarts = 8

        while calls_made < budget:
            # Deadline check
            if time.time() > deadline:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Accept
                x = y
                fx = fy
                best_x = x.copy()
                best_f = fx
                no_improve_count = 0
                # Gradually increase sigma on success
                sigma = min(sigma * 1.1, 3.0)
            else:
                no_improve_count += 1
                # Gradually decrease sigma on failure
                sigma = max(sigma * 0.9, 0.01)

            # Smart restart: only if significantly stuck and we have restarts left
            if no_improve_count >= stagnation_limit and restart_count < max_restarts:
                restart_count += 1
                # Perturb from the global best with moderate noise
                noise_scale = 1.0 + 0.5 * restart_count
                x = np.clip(best_x + noise_scale * rng.standard_normal(dim), -5, 5)
                fx = func(x)
                calls_made += 1
                no_improve_count = 0
                sigma = 0.8
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                # Tighten stagnation limit as we use more restarts
                stagnation_limit = max(15, stagnation_limit - 3)

            # If no restarts left, just keep going with current sigma
            if no_improve_count > 100 and restart_count >= max_restarts:
                # Final phase: very small steps to polish
                sigma = 0.05

        return best_x, best_f
# EVOLVE-BLOCK-END
