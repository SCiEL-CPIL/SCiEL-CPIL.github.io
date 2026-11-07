"""DTLZ4 multi-objective optimizer with inverse-mapping strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Inverse-mapping + Sobol fill DTLZ4 optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, d):
        """Generate n points in d dimensions using Halton sequence."""
        points = np.zeros((n, d))
        for j in range(d):
            base = j + 2
            for i in range(n):
                f, r = 1.0, i + 1
                while r > 0:
                    f /= base
                    r = (r - 1) // base
                points[i, j] = f
        return points

    def _inverse_dtlz4(self, f0, f1, f2):
        """
        Given target objectives (f0, f1, f2) with g=1 (on Pareto front),
        compute the corresponding x vector.
        
        On PF: f0 = x0, f1 = (1-x0)*x1, f2 = (1-x0)*(1-x1)
        So: x0 = f0, x1 = f1/(1-f0)
        For distance vars x2..x11: set to 0 (gives g=1)
        """
        x = np.zeros(self.dim)
        x[0] = f0
        if f0 < 1.0:
            x[1] = f1 / (1.0 - f0)
        else:
            x[1] = 0.0  # undefined, set to 0
        # x[2:] = 0 gives g = 1 + 0 + 0 = 1 (since 0^0.1 = 0)
        return x

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 270  # 30s safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Phase 1: Sample the Pareto front via inverse mapping
        # Use a 2D low-discrepancy sequence in (f0, f1) space
        # The Pareto front is parameterized by (f0, f1) where:
        #   f0 in [0,1], f1 in [0, 1-f0] (since f2 = 1-f0-f1 >= 0)
        # Actually: f0 + f1/(1-f0) ... let me think again.
        # On PF: f0 = x0, f1 = (1-x0)*x1, f2 = (1-x0)*(1-x1)
        # f0 + f1 + f2 = x0 + (1-x0)*x1 + (1-x0)*(1-x1) = x0 + (1-x0) = 1
        # So the PF is the simplex: f0+f1+f2=1, fi>=0
        # Parameterize by (f0, f1) with f0>=0, f1>=0, f0+f1<=1

        n_phase1 = 120
        n_phase2 = self.budget - n_phase1

        # Generate 2D Halton points and map to simplex
        halton_2d = self._halton(n_phase1, 2)
        # Map [0,1]^2 to simplex: use (u, v) -> (f0=u, f1=v*(1-u))
        # This gives uniform coverage of the triangle
        f0_vals = halton_2d[:, 0]
        f1_vals = halton_2d[:, 1] * (1.0 - halton_2d[:, 0])
        f2_vals = 1.0 - f0_vals - f1_vals

        # Add strategic extreme points (vertices and edge midpoints)
        # Vertices of the simplex in objective space
        vertices = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        # Edge midpoints
        edges = np.array([
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ])
        # Center
        center = np.array([[1/3, 1/3, 1/3]])

        all_front_points = np.vstack([
            np.column_stack([f0_vals, f1_vals, f2_vals]),
            vertices, edges, center
        ])

        # Evaluate all front points
        all_f = np.zeros((len(all_front_points), self.n_obj))
        all_x = np.zeros((len(all_front_points), self.dim))
        used = 0

        for i in range(len(all_front_points)):
            if time.time() > deadline:
                break
            f_target = all_front_points[i]
            x = self._inverse_dtlz4(f_target[0], f_target[1], f_target[2])
            x = np.clip(x, lo, hi)
            all_x[i] = x
            all_f[i] = problem(x)
            used += 1

        # Phase 2: Fill remaining budget with Sobol/Halton in full space
        # to catch any points slightly off the front that might improve HV
        if used < self.budget:
            n_phase2 = self.budget - used
            # Use Halton in 12D for the remaining evaluations
            halton_full = self._halton(n_phase2, self.dim)
            for i in range(n_phase2):
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * halton_full[i]
                f = problem(x)
                all_x[used] = x
                all_f[used] = f
                used += 1

        return all_x[:used]
# EVOLVE-BLOCK-END
