"""Optimized search for BBOB f15 (rotated Rastrigin)."""
import numpy as np
import time


class Optimizer:
    """Hybrid lattice-aware optimizer for rotated Rastrigin f15."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        def time_ok():
            return time.time() < deadline

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Phase 1: Lattice search using Rastrigin periodicity
        # Rastrigin: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        # Period is 2*pi in each coordinate. Minima at integer multiples of 2*pi.
        # In rotated frame, the lattice points are at R^T * (2*pi * k) for integer k.
        # We don't know R, but we can search a grid in the original space.
        # Since the box is [-5,5]^5 and 2*pi ≈ 6.28, there's roughly 1-2 periods per axis.
        # Generate candidate lattice points: for each axis, try offsets that align with 2*pi grid.
        # Without knowing the rotation, we use a dense grid + local refinement.

        # Strategy: 
        # 1. Evaluate a coarse grid to find the best region
        # 2. Do local (1+1)-ES refinement from multiple starting points
        # 3. Use the Rastrigin structure: near a local minimum, the function is quadratic
        #    so we can do a few Newton-like steps using finite differences

        # Coarse grid: 3^5 = 243 points
        grid_points = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            x = np.array([
                                -5 + 10 * i / 2,
                                -5 + 10 * j / 2,
                                -5 + 10 * k / 2,
                                -5 + 10 * l / 2,
                                -5 + 10 * m / 2,
                            ])
                            grid_points.append(x)

        # Also add a finer grid: 4^5 = 1024 points - too many, use 3^5 = 243
        # Let's do a two-stage: first 3^5, then refine top few

        # Evaluate coarse grid
        results = []
        for x in grid_points:
            if not time_ok():
                break
            fx = func(x)
            results.append((fx, x))

        results.sort(key=lambda t: t[0])

        # Take top 5 points for local refinement
        top_k = 5
        best_x = results[0][1].copy()
        best_f = results[0][0]

        calls_made = len(results)

        # Phase 2: Local (1+1)-ES from each of the top points
        # Use remaining budget for local search
        remaining = budget - calls_made
        if remaining <= 0:
            return best_x, best_f

        # Allocate remaining budget among top_k points
        # Give more budget to better points
        allocs = []
        total_weight = sum(1 / (i + 1) for i in range(top_k))
        for i in range(top_k):
            w = (1 / (i + 1)) / total_weight
            allocs.append(int(remaining * w))
        # Ensure sum doesn't exceed remaining
        while sum(allocs) > remaining:
            allocs[-1] -= 1
        while sum(allocs) < remaining:
            allocs[0] += 1

        for idx in range(top_k):
            if not time_ok():
                break
            x = results[idx][1].copy()
            fx = results[idx][0]
            sigma = 0.5
            n_iter = allocs[idx]

            for _ in range(n_iter):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85
                if sigma < 0.01:
                    sigma = 0.01

            if fx < best_f:
                best_x, best_f = x, fx

        # Phase 3: Final polish - very small steps
        if time_ok():
            sigma = 0.05
            for _ in range(50):
                if not time_ok():
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < best_f:
                    best_x, best_f = y, fy

        return best_x, best_f
