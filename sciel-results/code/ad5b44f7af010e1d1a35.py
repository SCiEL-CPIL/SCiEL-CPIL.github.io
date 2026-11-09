"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive sigma, designed for BBOB-style f17."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _run_es(self, x, sigma, budget_remaining, rng, func, deadline):
        """Run a single (1+1)-ES run. Returns (best_x, best_f)."""
        n = self.dim
        x = np.array(x, dtype=float)
        sigma = float(sigma)
        best_x = x.copy()
        best_f = func(x)
        f_cur = best_f
        budget_used = 1
        stall_count = 0
        stall_limit = 30

        while budget_used < budget_remaining:
            if time.time() > deadline:
                break
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            budget_used += 1

            if fy < f_cur:
                x = y
                f_cur = fy
                if fy < best_f:
                    best_f = fy
                    best_x = x.copy()
                stall_count = 0
                # Increase sigma on success
                sigma *= 1.15
            else:
                # Decrease sigma on failure
                sigma *= 0.92
                stall_count += 1
                # If sigma gets too small and we're stalling, break out
                if sigma < 1e-6 or stall_count > stall_limit:
                    break

            # Clamp sigma to reasonable range
            sigma = np.clip(sigma, 1e-8, 20.0)

        return best_x, best_f

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill
        total_budget = self.budget
        dim = self.dim
        n_evals = 0
        global_best_x = None
        global_best_f = np.inf

        rng = np.random.default_rng(12345)

        # Strategy: Multi-restart (1+1)-ES
        # Phase 1: Initial exploration with random starts and short ES runs
        # Phase 2: Refinement from the best points found

        # First, evaluate origin (often good for these functions)
        x0 = np.zeros(dim)
        f0 = func(x0)
        n_evals += 1
        global_best_x = x0.copy()
        global_best_f = f0

        # Define restart points
        # We'll use a mix of random points and points around the current best
        restart_points = []

        # Add some random initial points
        n_random_starts = 5
        for _ in range(n_random_starts):
            if n_evals >= total_budget or time.time() > deadline:
                break
            x_rand = rng.uniform(-5, 5, dim)
            restart_points.append(x_rand)

        # Add points in a small neighborhood of the current best (if not origin)
        # This helps with local refinement
        n_local_starts = 3
        for _ in range(n_local_starts):
            if n_evals >= total_budget or time.time() > deadline:
                break
            x_local = global_best_x + 0.5 * rng.standard_normal(dim)
            x_local = np.clip(x_local, -5, 5)
            restart_points.append(x_local)

        # Allocate budget to each restart
        # Use a larger share for the first few restarts, then smaller
        remaining_budget = total_budget - n_evals
        if remaining_budget <= 0 or time.time() > deadline:
            return global_best_x, global_best_f

        # We'll do sequential restarts, but we need to track global best
        # and potentially restart from it if we find something better

        # Strategy:
        # 1. Do a few short exploratory runs (50-100 evals each)
        # 2. Then do a longer refinement run from the global best

        # Phase 1: Exploratory restarts
        n_exploratory = 3
        budget_per_exploratory = max(30, min(80, remaining_budget // (n_exploratory + 1)))

        for i in range(n_exploratory):
            if n_evals >= total_budget or time.time() > deadline:
                break
            if i < len(restart_points):
                x_start = restart_points[i]
            else:
                x_start = rng.uniform(-5, 5, dim)

            # Run ES for a limited budget
            budget_for_this = min(budget_per_exploratory, total_budget - n_evals)
            if budget_for_this < 5:
                break

            best_x, best_f = self._run_es(
                x_start, 2.0, budget_for_this, rng, func, deadline
            )
            n_evals += budget_for_this

            if best_f < global_best_f:
                global_best_f = best_f
                global_best_x = best_x.copy()

        # Phase 2: Long refinement from the global best
        remaining_budget = total_budget - n_evals
        if remaining_budget > 20 and time.time() < deadline:
            # Use a smaller sigma for refinement
            sigma_refine = 0.5
            best_x, best_f = self._run_es(
                global_best_x, sigma_refine, remaining_budget, rng, func, deadline
            )
            n_evals += remaining_budget

            if best_f < global_best_f:
                global_best_f = best_f
                global_best_x = best_x.copy()

        # Final check: if we have budget left, try a few more small steps
        # This can help with final convergence
        remaining_budget = total_budget - n_evals
        if remaining_budget > 0 and time.time() < deadline:
            x = global_best_x.copy()
            f_cur = global_best_f
            sigma = 0.1  # Very small steps
            for _ in range(remaining_budget):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                n_evals += 1
                if fy < f_cur:
                    x = y
                    f_cur = fy
                    if fy < global_best_f:
                        global_best_f = fy
                        global_best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.95
                sigma = np.clip(sigma, 1e-8, 5.0)

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
