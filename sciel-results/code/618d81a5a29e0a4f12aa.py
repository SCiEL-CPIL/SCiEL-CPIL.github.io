"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES style optimizer for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start exploration (use ~25% of budget)
        explore_budget = max(10, budget // 4)
        candidates = []
        for i in range(explore_budget):
            if time.time() > deadline:
                break
            x = rng.uniform(-4, 4, n)
            fx = func(x)
            candidates.append((fx, x))

        # Sort by fitness, keep top few as starting points
        candidates.sort(key=lambda t: t[0])
        best_fx, best_x = candidates[0]

        # Phase 2: Adaptive (1+1)-ES with restarts from best starting point
        # Use remaining budget for exploitation
        remaining_budget = budget - len(candidates)
        if remaining_budget <= 0:
            return best_x, best_fx

        # CMA-ES-like approach with covariance adaptation
        x = best_x.copy()
        fx = best_fx
        sigma = 1.0
        # Simple covariance: start with identity, adapt based on successful steps
        C = np.eye(n)
        # Track recent successful directions for covariance update
        success_dirs = []
        stagnation_count = 0
        best_seen = fx
        restart_threshold = 30  # restart after 30 steps without improvement

        for step in range(remaining_budget):
            if time.time() > deadline:
                break

            # Generate step from adapted distribution
            # Use Cholesky-like approach: sample from standard normal, transform
            z = rng.standard_normal(n)
            # Simple covariance: use running average of outer products of successful directions
            if len(success_dirs) >= n:
                # Compute empirical covariance from last n successful directions
                recent = np.array(success_dirs[-n:])
                C = (recent.T @ recent) / n + 0.1 * np.eye(n)
                # Ensure positive definite
                C = (C + C.T) / 2
            # Transform z using C
            # Use simple square root via eigen decomposition (5x5 is fast)
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-6)
                step_vec = eigvecs @ (np.sqrt(eigvals) * z)
            except np.linalg.LinAlgError:
                step_vec = z

            y = np.clip(x + sigma * step_vec, -5, 5)
            fy = func(y)

            if fy < fx:
                # Success
                x, fx = y, fy
                success_dirs.append(step_vec)
                if len(success_dirs) > 50:
                    success_dirs = success_dirs[-50:]
                sigma *= 1.2
                stagnation_count = 0
                if fx < best_seen:
                    best_seen = fx
            else:
                # Failure
                sigma *= 0.85
                stagnation_count += 1

            # Periodic restart if stagnating
            if stagnation_count >= restart_threshold:
                # Restart from a random point near current best or a new random point
                if rng.random() < 0.5:
                    # Local restart
                    x = np.clip(best_x + rng.standard_normal(n) * 0.5, -5, 5)
                else:
                    # Global restart
                    x = rng.uniform(-4, 4, n)
                fx = func(x)
                sigma = 1.5
                success_dirs = []
                stagnation_count = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
