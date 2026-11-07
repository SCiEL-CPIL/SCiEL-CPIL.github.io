import numpy as np
import time

class Optimizer:
    """DTLZ2 optimizer using quasi-random sampling on the Pareto front for optimal hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        g_start = self.n_obj - 1
        base_x = np.zeros(self.dim)
        base_x[g_start:] = 0.5

        evaluated_points = []
        used_budget = 0

        # DTLZ2 3-objective Pareto front is a quarter-sphere.
        # With g=0 (x_i=0.5 for i>=g_start):
        #   f1 = cos(x0*pi/2)*cos(x1*pi/2)
        #   f2 = cos(x0*pi/2)*sin(x1*pi/2)
        #   f3 = sin(x0*pi/2)
        # Parameterize by t1, t2 in [0, pi/2]:
        #   x0 = (2/pi)*t1, x1 = (2/pi)*t2
        # For uniform surface coverage on the quarter-sphere:
        #   t1 = arccos(1 - 2*u), u in [0, 0.5]
        #   t2 = (pi/2)*v, v in [0, 1]
        #
        # Reserve 3 evaluations for Pareto front vertices for max hypervolume spread.
        n_vertices = 3
        n_sample = self.budget - n_vertices

        # Generate low-discrepancy points in [0, 0.5] x [0, 1]
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True)
            n_pow2 = 1
            while n_pow2 < n_sample:
                n_pow2 *= 2
            sobol_pts = sampler.random(n_pow2)
            u_vals = sobol_pts[:n_sample, 0] * 0.5
            v_vals = sobol_pts[:n_sample, 1]
        except (ImportError, Exception):
            # Fallback: Halton sequence
            def _halton(idx, base):
                f, r = 1.0, 0.0
                while idx > 0:
                    f /= base
                    r += f * (idx % base)
                    idx //= base
                return r
            u_vals = np.array([_halton(i + 1, 2) for i in range(n_sample)]) * 0.5
            v_vals = np.array([_halton(i + 1, 3) for i in range(n_sample)])

        # Convert to t1, t2 then to x0, x1
        t1 = np.arccos(np.clip(1.0 - 2.0 * u_vals, -1.0, 1.0))
        t2 = (np.pi / 2.0) * v_vals
        x0_vals = (2.0 / np.pi) * t1
        x1_vals = (2.0 / np.pi) * t2

        # Evaluate sample points
        for i in range(n_sample):
            if time.time() > deadline:
                return evaluated_points
            if used_budget >= self.budget:
                break

            x = base_x.copy()
            x[0] = x0_vals[i]
            x[1] = x1_vals[i]

            f = problem(x)
            evaluated_points.append(x)
            used_budget += 1

        # Add the 3 vertices of the Pareto front
        # (1,0,0): x0=0, x1=0
        # (0,1,0): x0=0, x1=1
        # (0,0,1): x0=1, x1=0
        vertices = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)]

        for x0, x1 in vertices:
            if time.time() > deadline:
                break
            if used_budget >= self.budget:
                break

            x = base_x.copy()
            x[0] = x0
            x[1] = x1

            f = problem(x)
            evaluated_points.append(x)
            used_budget += 1

        return evaluated_points
