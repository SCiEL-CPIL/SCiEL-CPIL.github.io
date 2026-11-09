"""Rewritten optimizer for BBOB-style f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(12345)

        # We'll track the global best
        best_x = None
        best_f = np.inf

        # Strategy: multiple restarts with different initializations
        # Phase 1: Latin Hypercube-like initial points (spread out)
        # Phase 2: (1+1)-ES from each point, allocating budget adaptively

        # Number of restarts: use a portion of budget for exploration
        # With 500 budget, do about 5-10 restarts
        n_restarts = 8
        budget_per_restart = budget // n_restarts

        # Generate diverse initial points using stratified sampling
        # For dim=5, create points that cover the space
        init_points = []
        # Center point
        init_points.append(np.zeros(dim))
        # Corner-like points (stratified)
        for i in range(n_restarts - 1):
            # Use quasi-random (Halton-like) or stratified
            # Simple approach: jittered grid
            p = np.array([rng.uniform(-4.5, 4.5) for _ in range(dim)])
            init_points.append(p)

        # Shuffle to avoid bias in ordering
        indices = rng.permutation(len(init_points))
        init_points = [init_points[i] for i in indices]

        # Track per-restart bests to decide allocation
        restart_bests = []

        for r_idx, x0 in enumerate(init_points):
            # Check time
            if time.time() > deadline - 30:
                break
            # Check remaining budget
            remaining = budget - sum(b for _, b in restart_bests)
            if remaining <= 0:
                break

            # Adaptive budget: give more to better restarts
            # For now, use proportional allocation based on early performance
            if r_idx == 0:
                alloc = min(budget_per_restart, remaining)
            else:
                # Simple: equal allocation for now, could be adaptive
                alloc = min(budget_per_restart, remaining)

            if alloc < 10:
                break

            # (1+1)-ES with 1/5 success rule
            x = np.clip(x0, -5, 5)
            fx = func(x)
            restart_bests.append((x.copy(), alloc))

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.5  # initial step size
            success_count = 0
            total_count = 0

            for step in range(alloc):
                # Time check
                if time.time() > deadline - 20:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_count += 1

                if fy < fx:
                    x = y
                    fx = fy
                    success_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.25
                else:
                    sigma *= 0.8

                # Adaptive sigma adjustment based on success rate
                if total_count >= 10:
                    rate = success_count / total_count
                    if rate > 0.3:
                        sigma *= 1.1
                    elif rate < 0.1:
                        sigma *= 0.9
                    success_count = 0
                    total_count = 0

        # If we have time and budget left, do a final fine-tuning from best point
        elapsed = time.time() - start_time
        if elapsed < 250 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            # Fine-grained search
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                if fy < fx:
                    x = y
                    fx = fy
                    best_f = fx
                    best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
