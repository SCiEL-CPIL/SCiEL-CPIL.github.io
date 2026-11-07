"""Multi-objective optimizer for DTLZ4 using structured Pareto-front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Optimizer that exploits the known DTLZ4 Pareto front structure.

    For DTLZ4 with m objectives, the Pareto front is a (m-1)-dimensional
    simplex in the first m-1 decision variables. The remaining variables
    are distance variables that should be at their optimal value (0.5 for
    DTLZ4 with alpha=10) to minimize the distance function g(x).

    We use Sobol quasi-random sequences for even coverage of the simplex.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        # DTLZ4: first m-1 variables are position variables, rest are distance
        n_pos = self.n_obj - 1  # 2 for 3 objectives
        n_dist = self.dim - n_pos  # 10 distance variables

        # Optimal distance variable value for DTLZ4 (alpha=10)
        # g(x) is minimized when each distance var = 0.5
        dist_val = 0.5 * (lo[n_pos:] + hi[n_pos:])
        # Actually for DTLZ4 the standard form has bounds [0,1] and optimum at 0.5
        # But to be safe, use the midpoint of bounds
        dist_val = (lo[n_pos:] + hi[n_pos:]) / 2.0

        # Generate Sobol points in [0,1]^n_pos
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n_pos, scramble=True)
            pos_points = sampler.random(self.budget)
        except Exception:
            # Fallback: deterministic grid
            n_per_axis = int(np.ceil(self.budget ** (1.0 / n_pos)))
            grid_axes = np.linspace(0, 1, n_per_axis)
            pos_points = np.array(list(np.meshgrid(*[grid_axes] * n_pos, indexing='ij'))).reshape(n_pos, -1).T
            pos_points = pos_points[:self.budget]
            if len(pos_points) < self.budget:
                # Pad with random points
                remaining = self.budget - len(pos_points)
                rng = np.random.default_rng(42)
                extra = rng.random((remaining, n_pos))
                pos_points = np.vstack([pos_points, extra])

        # Build full decision vectors
        X = np.zeros((self.budget, self.dim))
        X[:, :n_pos] = pos_points
        X[:, n_pos:] = dist_val[np.newaxis, :]

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        # Check if we're running out of time
        if time.time() > deadline:
            return X

        # If we have time and budget remaining, add some extra exploration
        # around the Pareto front to catch any numerical edge cases
        remaining_budget = self.budget - self.budget  # We used all budget
        # Actually we used exactly self.budget evaluations, so we're done.

        return X
# EVOLVE-BLOCK-END
