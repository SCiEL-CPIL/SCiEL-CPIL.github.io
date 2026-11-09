"""Optimizer for BBOB f23 (Katsuura) with multi-start and adaptive (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size and periodic restarts for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf

        # Phase 1: Latin Hypercube sampling for fast early space-filling progress
        n_lhs = min(50, budget // 10)
        lhs_points = self._latin_hypercube(rng, n_lhs, dim)

        # Store all evaluations for potential multi-start
        eval_points = []
        eval_fvals = []

        for i in range(n_lhs):
            if time.time() > deadline:
                break
            x = lhs_points[i]
            fx = func(x)
            eval_points.append(x)
            eval_fvals.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES with periodic restarts from global best
        # Katsuura has dense local minima; periodic restarts with larger sigma help escape
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        remaining_budget = budget - n_lhs

        # Track progress for adaptive restarts
        no_improve_count = 0
        restart_interval = 80  # restart every 80 steps if no improvement
        restart_sigma_multiplier = 2.0

        for i in range(remaining_budget):
            if time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)

            if fy < fx:
                x = y
                fx = fy
                no_improve_count = 0
                # Increase sigma on success but more conservatively
                sigma *= 1.1
            else:
                no_improve_count += 1
                # Decrease sigma on failure
                sigma *= 0.9

            # Keep track of global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Periodic restart: if no improvement for restart_interval steps,
            # restart from global best with larger sigma to escape local minima
            if no_improve_count >= restart_interval:
                x = best_x.copy()
                fx = best_f
                sigma = max(sigma * restart_sigma_multiplier, 0.5)
                no_improve_count = 0
                # Reduce restart interval over time to try more restarts
                restart_interval = max(40, restart_interval - 10)

            # Clamp sigma to reasonable range
            sigma = np.clip(sigma, 0.005, 5.0)

        # Phase 3: If budget remains, try a few more random starts
        calls_made = n_lhs + remaining_budget
        if calls_made < budget and time.time() < deadline - 10:
            for _ in range(min(5, budget - calls_made)):
                if time.time() > deadline:
                    break
                x0 = rng.uniform(-4, 4, dim)
                fx0 = func(x0)
                if fx0 < best_f:
                    best_f = fx0
                    best_x = x0.copy()

        return best_x, best_f

    def _latin_hypercube(self, rng, n, dim):
        """Generate Latin Hypercube points in [-5, 5]^dim."""
        points = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            points[:, d] = -5 + 10 * (perm + rng.random(n)) / n
        return points
# EVOLVE-BLOCK-END
