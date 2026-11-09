"""Optimizer for BBOB f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive sigma for f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s kill

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global exploration with multiple restarts
        # Phase 2: Local refinement on the best point found

        # Strategy: Use a portion of budget for multi-start exploration,
        # then refine the best solution found.

        # Split budget: ~40% for exploration, ~60% for refinement
        n_restarts = 10
        calls_per_restart = max(1, (budget // 3) // n_restarts)  # ~1/3 budget for exploration
        refinement_budget = budget - n_restarts * calls_per_restart - 1

        best_x = None
        best_f = np.inf
        call_count = 0

        # Deterministic starting points + random
        # Use a quasi-random spread for initial points
        # Halton-like or just grid corners + random

        # Create diverse starting points
        start_points = []
        # Corners and center
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        for m in range(2):
                            pt = np.array([
                                -5 + 10 * i,
                                -5 + 10 * j,
                                -5 + 10 * k,
                                -5 + 10 * l,
                                -5 + 10 * m,
                            ])
                            start_points.append(pt)
        # Center
        start_points.append(np.zeros(dim))
        # Add random points
        while len(start_points) < n_restarts:
            start_points.append(rng.uniform(-5, 5, dim))

        # Shuffle for diversity
        idx = rng.permutation(len(start_points))
        start_points = [start_points[i] for i in idx[:n_restarts]]

        for sp in start_points:
            if call_count >= budget - 1:
                break
            if time.time() > deadline - 10:
                break

            x = np.clip(sp.copy(), -5, 5)
            fx = func(x)
            call_count += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.5
            # Run (1+1)-ES from this point
            for _ in range(calls_per_restart - 1):
                if call_count >= budget - 1:
                    break
                if time.time() > deadline - 10:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                call_count += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                    # If we've been stuck for a while, try a bigger jump
                    if sigma < 0.1:
                        sigma = 0.5

        # Phase 2: Refinement on the best point found
        if best_x is not None and refinement_budget > 0:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5

            for _ in range(refinement_budget):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                call_count += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9

                    # Adaptive sigma reset if stuck
                    if sigma < 0.05:
                        sigma = 0.3

        # Ensure we don't exceed budget
        # The above loops already check call_count

        return best_x, best_f
# EVOLVE-BLOCK-END
