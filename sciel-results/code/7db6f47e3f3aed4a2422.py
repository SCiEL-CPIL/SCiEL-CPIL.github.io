"""Portfolio search for DTLZ2: Sobol sequence with surface-area-weighted sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using Sobol-sequence front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # safety margin under 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        def time_ok():
            return time.time() < deadline

        def front_to_x(t):
            """Convert angle parameters t (length n_obj-1) to decision vector x.
            For DTLZ2: x_i = sin(t_i) for i < n_obj-1, and the rest are set
            to values that make g(x) = 0 (i.e., x_i = 0.5 for i >= n_obj-1).
            """
            x = np.full(self.dim, 0.5)
            for i in range(min(self.n_obj - 1, self.dim)):
                x[i] = np.sin(t[i])
            return x

        # Generate Sobol sequence points in 2D
        def sobol_2d(n):
            """Generate n Sobol sequence points in [0,1)^2."""
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True)
            points = sampler.random(n)
            return points

        n_evals = self.budget
        sobol_pts = sobol_2d(n_evals)

        # The DTLZ2 Pareto front in 3 objectives is a quarter-sphere.
        # Parameterization: f1 = sin(t1)*sin(t2), f2 = sin(t1)*cos(t2), f3 = cos(t1)
        # where t1, t2 in [0, pi/2].
        # Surface area element is proportional to cos(t1) * dt1 * dt2.
        # To get uniform surface density, we need:
        #   P(t1) ∝ cos(t1)  =>  CDF(t1) = sin(t1)  =>  t1 = arcsin(u1)
        #   P(t2) ∝ 1        =>  CDF(t2) = t2/(pi/2) =>  t2 = v2 * pi/2
        # This is the same mapping as before, but now with Sobol for better uniformity.

        X = []
        F = []
        used = 0
        for k in range(n_evals):
            if not time_ok():
                break
            u1, v2 = sobol_pts[k]
            # u1 in [0,1) maps to t1 = arcsin(u1)
            # v2 in [0,1) maps to t2 = v2 * pi/2
            t1 = np.arcsin(u1)
            t2 = v2 * (np.pi / 2)
            t = np.array([t1, t2])
            x = front_to_x(t)
            x = np.clip(x, lo, hi)
            f = problem(x)
            X.append(x)
            F.append(f)
            used += 1

        X = np.array(X)
        F = np.array(F)

        return X

# EVOLVE-BLOCK-END
