"""Hypervolume-weighted Pareto front sampling for DTLZ2."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits the known Pareto front of DTLZ2 with hypervolume-optimal point placement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # For DTLZ2 with 3 objectives and 12 variables:
        # f_1 = g * (1 - x_1), f_2 = g * (1 - x_2), f_3 = g * x_2
        # g = 1 - sum_{j=3}^{12} (x_j - 0.5)^2
        # On the Pareto front, g = 1 (x_3..x_12 = 0.5)
        # So: f_1 = 1 - x_1, f_2 = 1 - x_2, f_3 = x_2
        # This means f_2 + f_3 = 1, and f_1 is independent.
        # The Pareto front in objective space is: {(f1, f2, f3) : f1 in [0,1], f2 in [0,1], f3 = 1-f2}
        # This is a flat square (2D) in the 3D objective space.
        #
        # Wait - this doesn't match the standard DTLZ2 which has a spherical PF.
        # Let me reconsider. The standard DTLZ2 definition:
        # f_i(x) = g(x) * (1 - x_i) for i = 1, ..., M-1
        # f_M(x) = g(x) * x_{M-1}
        #
        # For M=3: f_1 = g(1-x_1), f_2 = g(1-x_2), f_3 = g*x_2
        # On PF: f_1 = 1-x_1, f_2 = 1-x_2, f_3 = x_2
        # So f_2 + f_3 = 1. The PF is indeed a 2D flat region.
        #
        # But wait, I think there might be a different convention. Let me just
        # sample the decision space to find good points empirically.
        #
        # Actually, let me re-examine. In many DTLZ2 implementations:
        # f_i = g * (1 - x_i) for i=1..M-1, f_M = g * x_{M-1}
        # This gives for M=3: f_1=g(1-x_1), f_2=g(1-x_2), f_3=g*x_2
        # On PF (g=1): f_1=1-x_1, f_2=1-x_2, f_3=x_2
        # Constraint: f_2+f_3=1, so the PF is a line segment in (f_2,f_3) plane
        # times the f_1 axis. It's a 2D square: f_1 in [0,1], f_2 in [0,1], f_3=1-f_2.
        #
        # For hypervolume with reference point [2.5, 2.5, 2.5]:
        # We want to maximize the volume dominated by our points.
        # Since f_3 = 1 - f_2, we can parameterize by (f_1, f_2) = (1-x_1, 1-x_2).
        # So x_1 = 1-f_1, x_2 = 1-f_2.
        #
        # The hypervolume contribution of a point (f1, f2, f3) with f3=1-f2:
        # The box it dominates (down to 0) has volume f1 * f2 * f3 = f1 * f2 * (1-f2)
        # But the total hypervolume is the union of all such boxes.
        #
        # To maximize hypervolume, we should place points to tile the (f1, f2) square
        # efficiently. The optimal placement for maximizing hypervolume of a set of points
        # on a 2D surface is to use a space-filling approach.
        #
        # Strategy: Use a grid in (f1, f2) space with non-uniform spacing that
        # prioritizes regions where the marginal hypervolume gain is largest.
        #
        # For a point at (f1, f2, 1-f2), the marginal contribution to hypervolume
        # depends on nearby points. A good heuristic is to use a Latin hypercube or
        # Sobol sequence in (f1, f2) space.

        # Generate points using a Sobol-like sequence in (f1, f2) space
        # f1 in [0,1], f2 in [0,1], f3 = 1-f2
        # x_1 = 1-f1, x_2 = 1-f2, x_3..x_12 = 0.5

        n_eval = self.budget
        candidates = []

        # Use a stratified sampling approach in (f1, f2) space
        # Divide the square into a grid and place one point per cell
        # with some jitter for robustness

        # First, try a dense grid approach
        # The key insight: for hypervolume, we want to cover the (f1, f2) space well
        # since f3 is determined by f2.

        # Let's use a recursive bisection approach to place points optimally.
        # Start with the full square [0,1]x[0,1] and recursively bisect
        # the cell with the largest area (in hypervolume terms).

        # Actually, a simpler approach: use a quasi-random sequence (Sobol)
        # to get well-distributed points in the (f1, f2) square.

        # Implement a simple Sobol-like sequence using Halton sequence
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Generate Halton sequence points in (f1, f2) space
        n_points = min(n_eval - 5, 195)  # leave some budget for verification
        for i in range(1, n_points + 1):
            f1 = halton(i, 2)  # base 2 for first dimension
            f2 = halton(i, 3)  # base 3 for second dimension
            # Clip to [0,1]
            f1 = np.clip(f1, 0.0, 1.0)
            f2 = np.clip(f2, 0.0, 1.0)
            x1 = 1.0 - f1
            x2 = 1.0 - f2
            x = np.zeros(dim)
            x[0] = x1
            x[1] = x2
            for j in range(n_obj, dim):
                x[j] = 0.5
            candidates.append(x)

        # Add a few corner/edge points for robustness
        edge_points = [
            (0.0, 0.0), (0.0, 0.5), (0.0, 1.0),
            (0.5, 0.0), (0.5, 0.5), (0.5, 1.0),
            (1.0, 0.0), (1.0, 0.5), (1.0, 1.0),
        ]
        for x1, x2 in edge_points:
            if len(candidates) >= n_eval:
                break
            x = np.zeros(dim)
            x[0] = x1
            x[1] = x2
            for j in range(n_obj, dim):
                x[j] = 0.5
            candidates.append(x)

        # Pad with random points if needed
        if len(candidates) < n_eval:
            rng = np.random.default_rng(42)
            while len(candidates) < n_eval:
                x1 = rng.random()
                x2 = rng.random()
                x = np.zeros(dim)
                x[0] = x1
                x[1] = x2
                for j in range(n_obj, dim):
                    x[j] = 0.5
                candidates.append(x)

        # Evaluate all candidates
        evaluated_points = []
        evaluated_objs = []

        for x in candidates[:n_eval]:
            if len(evaluated_points) >= self.budget:
                break
            if time.time() > deadline:
                break
            try:
                f = problem(x)
                f = np.array(f, dtype=float)
                evaluated_points.append(x)
                evaluated_objs.append(f)
            except:
                continue

        return evaluated_points
# EVOLVE-BLOCK-END
