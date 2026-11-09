"""Coarse-to-fine Halton grid search with local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic coarse-to-fine Halton grid with local (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # Halton sequence generator
        def halton_sequence(index, base):
            q = 1.0
            r = 0.0
            while index > 0:
                q /= base
                r += q * (index % base)
                index //= base
            return r

        primes = [2, 3, 5, 7, 11]

        def generate_halton_points(n, offset=0):
            """Generate n Halton points in [0,1)^dim, scaled to [-5,5]^dim."""
            pts = np.zeros((n, dim))
            for i in range(n):
                for j in range(dim):
                    pts[i, j] = halton_sequence(i + 1 + offset, primes[j % len(primes)])
            return pts * 10.0 - 5.0

        # Phase 1: Coarse Halton grid
        # Use 100 points for initial coverage
        coarse_points = generate_halton_points(100)
        for pt in coarse_points:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # Phase 2: Coarse-to-fine refinement
        # Refine around the best point found, halving the step size each round
        # We'll do up to 4 refinement rounds
        center = best_x.copy()
        step_size = 2.0  # Initial refinement radius

        for round_num in range(4):
            if calls >= budget:
                break
            if time.time() > deadline:
                break

            # Number of points in this refinement round
            n_refine = 50

            # Generate points in a box around center with radius step_size
            # Use Halton points scaled to [-step_size, step_size] + center
            refine_points = np.zeros((n_refine, dim))
            for i in range(n_refine):
                for j in range(dim):
                    # Generate in [0,1) then scale to [-step_size, step_size]
                    val = halton_sequence(i + 1 + round_num * 1000, primes[j % len(primes)])
                    refine_points[i, j] = center[j] + (val * 2.0 - 1.0) * step_size

            for pt in refine_points:
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                pt = np.clip(pt, -5, 5)
                f = func(pt)
                calls += 1
                if f < best_f:
                    best_f = f
                    best_x = pt.copy()
                    center = pt.copy()

            # Halve the step size for next round
            step_size *= 0.5

        # Phase 3: Local (1+1)-ES refinement around the best point
        x = best_x.copy()
        fx = best_f
        sigma = 0.3  # Start with a small step for local refinement

        success_window = []
        window_size = 10
        sigma_min = 0.001
        sigma_max = 1.0

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            # Track success
            success = fy < fx
            success_window.append(success)
            if len(success_window) > window_size:
                success_window.pop(0)

            # 1/5th success rule
            if len(success_window) >= window_size:
                success_rate = sum(success_window) / len(success_window)
                if success_rate > 0.3:
                    sigma = min(sigma * 1.15, sigma_max)
                elif success_rate < 0.1:
                    sigma = max(sigma * 0.75, sigma_min)

            # Update current solution
            if success:
                x, fx = y, fy

            # Update global best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
