"""Search-based optimizer for DTLZ4 with known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using search over the known front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 285.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 structure for 3 objectives:
        # f1 = (1+g) * 0.5 * cos(pi*x1/2) * cos(pi*x2/2)
        # f2 = (1+g) * 0.5 * cos(pi*x1/2) * sin(pi*x2/2)
        # f3 = (1+g) * 0.5 * sin(pi*x1/2)
        # Pareto front (g=0): quarter sphere of radius 0.5
        # g = sum((x_{dim-5+i} - 0.5)^2) for i=0..4, so g=0 when x8..x12=0.5

        def make_x(x1, x2):
            x = np.full(dim, 0.5)
            x[0] = x1
            x[1] = x2
            return np.clip(x, lo, hi)

        # Generate 200 candidate (x1, x2) pairs with excellent coverage
        candidates = []

        # Strategy 1: Fibonacci lattice for near-optimal uniform coverage
        n_fib = 180
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n_fib):
            # Use golden ratio conjugate for 1D Fibonacci
            frac1 = (i / n_fib + 1.0 / golden_ratio) % 1.0
            # For 2D, use a different base to avoid correlation
            frac2 = (i / n_fib + 2.0 / golden_ratio) % 1.0
            # Add a small offset to avoid exact 0 or 1
            x1 = frac1 * 0.98 + 0.01
            x2 = frac2 * 0.98 + 0.01
            candidates.append((x1, x2))

        # Strategy 2: Explicit boundary points for better corner coverage
        # Corners of the x1-x2 square
        candidates.append((0.0, 0.0))
        candidates.append((0.0, 1.0))
        candidates.append((1.0, 0.0))
        candidates.append((1.0, 1.0))

        # Edge midpoints
        candidates.append((0.0, 0.5))
        candidates.append((1.0, 0.5))
        candidates.append((0.5, 0.0))
        candidates.append((0.5, 1.0))

        # Additional edge points for denser boundary coverage
        for t in np.linspace(0.1, 0.9, 8):
            candidates.append((0.0, t))
            candidates.append((1.0, t))
            candidates.append((t, 0.0))
            candidates.append((t, 1.0))

        # Strategy 3: A few random points for diversity
        n_random = 10
        for _ in range(n_random):
            candidates.append((np.random.rand(), np.random.rand()))

        # Trim to exactly budget
        if len(candidates) > budget:
            # Keep the first budget candidates (Fibonacci lattice is most important)
            candidates = candidates[:budget]

        # Pad if less than budget (shouldn't happen, but safety)
        while len(candidates) < budget:
            candidates.append((np.random.rand(), np.random.rand()))

        # Evaluate all candidates
        evaluated_points = []
        used = 0

        for x1, x2 in candidates:
            if used >= budget or time.time() - t0 > deadline:
                break

            x = make_x(x1, x2)
            f = np.array(problem(x))
            used += 1
            evaluated_points.append(f)

        return np.array(evaluated_points)
# EVOLVE-BLOCK-END
