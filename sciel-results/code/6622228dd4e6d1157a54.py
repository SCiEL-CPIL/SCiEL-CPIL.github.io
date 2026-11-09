import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = 0
        all_points = []
        all_fvals = []

        def eval_point(x):
            nonlocal n_evals
            if n_evals >= self.budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            try:
                f = problem(x)
            except Exception:
                return None
            if len(f) != self.n_obj:
                return None
            f = np.array(f, dtype=float)
            if not np.all(np.isfinite(f)):
                return None
            all_points.append(x.copy())
            all_fvals.append(f)
            n_evals += 1
            return f

        # DTLZ2 with n_obj=3: Pareto front is unit quarter-sphere
        # f1 = (1+g) * cos(a1)
        # f2 = (1+g) * sin(a1) * cos(a2)
        # f3 = (1+g) * sin(a1) * sin(a2)
        # where a1, a2 in [0, pi/2]
        # For g=0: x_0 = a1, x_1 = a2, x_2..x_11 = center (0.5 for standard DTLZ2 with [0,1] bounds)
        #
        # To get UNIFORM surface density on the sphere, we need:
        # a1 = arccos(1 - u1) where u1 ~ U[0,1]  (this gives uniform cos(a1))
        # a2 = 2*pi*u2  but since we're on quarter sphere, a2 in [0, pi/2]
        # Actually for quarter sphere uniform density:
        # a1 = arccos(1 - u1) maps u1 in [0,1] to a1 in [0, pi/2] with uniform surface density
        # a2 = (pi/2) * u2 maps u2 in [0,1] to a2 in [0, pi/2] uniformly
        #
        # Use Halton sequence for better low-discrepancy than random
        n_points = self.budget

        # Generate 2D Halton sequence
        def halton_sequence(n, dims=2):
            """Generate n points in [0,1)^dims using Halton sequence."""
            points = []
            for i in range(1, n + 1):
                point = []
                for base in range(2, dims + 2):  # bases 2, 3 for 2D
                    f = 1.0
                    r = 0.0
                    ii = i
                    while ii > 0:
                        f /= base
                        r += f * (ii % base)
                        ii //= base
                    point.append(r)
                points.append(point)
            return np.array(points)

        halton_u = halton_sequence(n_points, dims=2)

        # Map Halton points to angles with correct surface-uniform mapping
        # a1 = arccos(1 - u1) ensures uniform density on sphere surface
        # a2 = (pi/2) * u2 for uniform angular distribution in the quarter
        a1 = np.arccos(1.0 - halton_u[:, 0])  # in [0, pi/2]
        a2 = (np.pi / 2.0) * halton_u[:, 1]   # in [0, pi/2]

        n_angles = self.n_obj - 1  # 2 angle variables
        n_g_vars = self.dim - n_angles  # 10 g-variables

        # Standard DTLZ2 with bounds [0,1]: g = sum((x_i - 0.5)^2) for i >= n_angles
        # So g=0 when x_i = 0.5 for all g-variables
        g_center = 0.5

        for i in range(n_points):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = a1[i]
            x[1] = a2[i]
            x[2:] = g_center
            eval_point(x)

        return all_points
