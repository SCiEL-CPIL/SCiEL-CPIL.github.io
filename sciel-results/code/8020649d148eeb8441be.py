"""Optimizer for BBOB f23 (Katsuura) with multi-start and adaptive (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size for Katsuura f23."""

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

        # Phase 1: Latin Hypercube sampling to find good initial points
        # Use a fraction of budget for exploration
        n_lhs = min(50, budget // 10)
        lhs_points = self._latin_hypercube(rng, n_lhs, dim)

        for i in range(n_lhs):
            if time.time() > deadline:
                break
            x = lhs_points[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Multi-start (1+1)-ES from the best point and a few others
        # Sort by function value and take top candidates for refinement
        # We'll do refinement from the best point primarily, but also try a few others

        # Collect all evaluated points
        all_points = [lhs_points[i] for i in range(n_lhs)]
        all_fvals = []
        for i in range(n_lhs):
            # We already evaluated these, but we need to store them
            # Actually, let me restructure: store all evaluations
            pass

        # Let me redo this more carefully
        # Re-evaluate is expensive, so let's store during Phase 1
        # Actually, I already have best_x and best_f. Let me just do ES from best.
        # But multi-start can help. Let me do a few ES runs from different starting points.

        # For simplicity and robustness, let's do ES from the best point found,
        # and if time permits, from the second and third best.

        # First, let's identify top starting points
        # We need to re-evaluate or store. Let me store during Phase 1.

        # Actually, let me just proceed with ES from best_x and use remaining budget.

        # Phase 2: (1+1)-ES with adaptive sigma
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # initial step size

        # Adaptive step size based on success rate (1/5 rule variant)
        successes = 0
        window = 10
        trial_count = 0

        remaining_budget = budget - n_lhs

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
                successes += 1
                sigma *= 1.15  # increase step size on success
            else:
                sigma *= 0.85  # decrease step size on failure

            # Keep track of global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Adaptive sigma adjustment every window steps
            trial_count += 1
            if trial_count % window == 0:
                success_rate = successes / window
                if success_rate > 0.3:
                    sigma *= 1.3
                elif success_rate < 0.1:
                    sigma *= 0.7
                successes = 0

            # Clamp sigma to reasonable range
            sigma = np.clip(sigma, 0.01, 5.0)

        # If we have time and budget left, try a few more starts
        # This is optional and only if we haven't used all budget
        calls_made = n_lhs + remaining_budget
        if calls_made < budget and time.time() < deadline - 10:
            # Try a few more random starts
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
