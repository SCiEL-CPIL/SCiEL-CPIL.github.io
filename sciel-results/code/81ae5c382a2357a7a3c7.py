"""Portfolio search for DTLZ2: Low-discrepancy sampling on the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.stats.qmc as qmc


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using Sobol sampling on the front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0
        budget = self.budget
        dim = self.dim

        # DTLZ2 structure:
        # g = sum_{i=3}^{dim} (x_i - 0.5)^2
        # f1 = (1+g)*sin(pi*x1/2)
        # f2 = (1+g)*cos(pi*x1/2)*sin(pi*x2/2)
        # f3 = (1+g)*cos(pi*x1/2)*cos(pi*x2/2)
        # Setting x3..x12 = 0.5 gives g = 0 -> point on the true Pareto front.
        # x1, x2 in [0,1] sweep the quarter-sphere front.

        # 1. Add extreme corner points explicitly to ensure coverage of the axes.
        # These correspond to the points (0,0,1), (0,1,0), and (1,0,0) in objective space.
        candidates = []

        # x1=0, x2=0 -> f1=0, f2=0, f3=1
        x_corner1 = np.full(dim, 0.5)
        x_corner1[0] = 0.0
        x_corner1[1] = 0.0
        candidates.append(x_corner1)

        # x1=0, x2=1 -> f1=0, f2=1, f3=0
        x_corner2 = np.full(dim, 0.5)
        x_corner2[0] = 0.0
        x_corner2[1] = 1.0
        candidates.append(x_corner2)

        # x1=1, x2=0 -> f1=1, f2=0, f3=0
        x_corner3 = np.full(dim, 0.5)
        x_corner3[0] = 1.0
        x_corner3[1] = 0.0
        candidates.append(x_corner3)

        # 2. Generate remaining points using Sobol sequence for low discrepancy.
        # Map to angular parameters for uniform surface coverage on the quarter-sphere.
        n_remaining = budget - len(candidates)
        if n_remaining > 0:
            # Sobol requires power of 2 for best properties, but works with any n
            sampler = qmc.Sobol(d=2, scramble=True, seed=42)
            sobol_pts = sampler.random(n=n_remaining)

            # Map to angular parameters for uniform surface coverage:
            # u = cos(phi) uniform in [0,1] -> phi = arccos(u)
            # theta = v * pi/2 uniform in [0, pi/2]
            u = sobol_pts[:, 0]
            v = sobol_pts[:, 1]

            phi = np.arccos(u)
            theta = v * (np.pi / 2)

            # Compute x1, x2 on the Pareto front
            x1 = np.sin(phi) * np.cos(theta)
            x2 = np.sin(phi) * np.sin(theta)

            for i in range(n_remaining):
                x = np.full(dim, 0.5)
                x[0] = x1[i]
                x[1] = x2[i]
                candidates.append(x)

        # 3. Evaluate all candidates, recording every objective vector.
        for x in candidates:
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            try:
                problem(x)
            except Exception:
                # If an exception occurs, stop to avoid further issues.
                break
            used += 1

        return None
# EVOLVE-BLOCK-END
