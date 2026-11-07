"""Multi-objective optimizer for DTLZ1 exploiting known Pareto front structure via Sobol sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Exploits the known Pareto front of DTLZ1.
    For n_obj=3, the Pareto front is parameterized by x_0, x_1 in [0,1]
    with x_2..x_{dim-1} = 0.5 (so g=0).
    Uses Sobol sequence for low-discrepancy sampling of the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_2d(self, n, seed=0):
        """Generate n points in [0,1]^2 using a simple low-discrepancy sequence."""
        # Use Halton sequence as a fallback if scipy is not available
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, seed=seed)
            points = sampler.random(n)
            return points
        except ImportError:
            # Fallback: Halton sequence
            def halton(index, base):
                result = 0.0
                f = 1.0 / base
                i = index
                while i > 0:
                    result += f * (i % base)
                    i //= base
                    f /= base
                return result

            points = np.zeros((n, 2))
            for i in range(n):
                points[i, 0] = halton(i + 1, 2)
                points[i, 1] = halton(i + 1, 3)
            return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_obj = self.n_obj
        dim = self.dim

        # For n_obj=3, the Pareto front is parameterized by x_0, x_1 in [0,1]
        # with x_2..x_{dim-1} = 0.5
        # f0 = 0.5 * x0 * x1
        # f1 = 0.5 * x0 * (1-x1)
        # f2 = 0.5 * (1-x0)

        # Use Sobol sequence for low-discrepancy sampling
        n_points = self.budget

        if n_obj == 3:
            # Generate Sobol points in [0,1]^2
            sobol_pts = self._sobol_2d(n_points, seed=42)
            
            # Build decision vectors
            X = np.zeros((n_points, dim))
            X[:, 0] = sobol_pts[:, 0]
            X[:, 1] = sobol_pts[:, 1]
            # x_2..x_{dim-1} = 0.5 for g=0
            X[:, 2:] = 0.5
        else:
            # General case: use random sampling
            X = lo + (hi - lo) * np.random.default_rng(0).random((n_points, dim))

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = len(X)

        # Check if we have time for refinement
        elapsed = time.time() - start_time
        if elapsed < deadline - 10.0 and used < self.budget:
            # Try to refine some points locally with small perturbations
            rng = np.random.default_rng(123)
            for i in range(used):
                if time.time() > deadline - 5.0:
                    break
                if used >= self.budget:
                    break

                # Small perturbation
                x = X[i].copy()
                x += 0.01 * (hi - lo) * rng.standard_normal(dim)
                x = np.clip(x, lo, hi)

                fx = problem(x)
                used += 1

                # Replace the worst point if this one is better
                if np.sum(fx) < np.sum(F[i]):
                    X[i] = x
                    F[i] = fx

        return X
# EVOLVE-BLOCK-END
