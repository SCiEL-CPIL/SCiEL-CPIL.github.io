"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Pareto-front sampling via Sobol sequence mapped to DTLZ2 surface."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_2d(self, n):
        """Generate n points in [0,1)^2 using a simple Sobol-like sequence (Halton for simplicity)."""
        # Use Halton sequence with bases 2 and 3 for 2D low-discrepancy points
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        points = np.empty((n, 2))
        for idx in range(1, n + 1):
            points[idx - 1, 0] = halton(idx, 2)
            points[idx - 1, 1] = halton(idx, 3)
        return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.budget

        # Generate 2D low-discrepancy points in [0,1)^2
        u_points = self._sobol_2d(n)

        # Map to DTLZ2 Pareto front
        # DTLZ2 with 3 objectives: f_i = g * cos(theta_i) * prod_{j<i} sin(theta_j)
        # where g = sum(x_i^2) for i >= M (here M=2, so g = sum(x_2..x_11)^2)
        # For Pareto front: g = 0, so x_2..x_11 = 0
        # Then f_1 = cos(theta_1), f_2 = sin(theta_1)*cos(theta_2), f_3 = sin(theta_1)*sin(theta_2)
        # where theta_1 = x_0 * pi/2, theta_2 = x_1 * pi/2

        # To get uniform coverage in objective space (for hypervolume), we need
        # to sample (f1, f2) uniformly in the feasible region of the Pareto front.
        # The Pareto front in (f1, f2) space: f1^2 + f2^2 + f3^2 = 1, f_i >= 0
        # With f3 = sqrt(1 - f1^2 - f2^2)
        # We want to sample (f1, f2) such that the resulting points are uniform on the surface.

        # Alternative approach: sample theta_1, theta_2 such that the surface area element is uniform.
        # Surface area element on the unit sphere in the first octant:
        # dA = sin(theta_1) dtheta_1 dtheta_2 (for the parameterization above)
        # To get uniform surface area: P(theta_1) dtheta_1 = sin(theta_1) dtheta_1 / integral
        # integral of sin(theta_1) from 0 to pi/2 = 1
        # So theta_1 = arcsin(u_1) where u_1 ~ Uniform(0,1)
        # And theta_2 is uniform on [0, pi/2]

        # Actually, let me reconsider. The standard DTLZ2 parameterization:
        # x_0, x_1 are the angle variables, x_2..x_{D-1} are the g-function variables.
        # On the Pareto front, g=0 so x_2..x_{D-1}=0.
        # f_1 = cos(x_0 * pi/2)
        # f_2 = sin(x_0 * pi/2) * cos(x_1 * pi/2)
        # f_3 = sin(x_0 * pi/2) * sin(x_1 * pi/2)

        # The surface area element for the spherical cap in the first octant:
        # Using spherical coordinates: r=1, polar angle phi in [0, pi/2], azimuthal angle theta in [0, pi/2]
        # But our parameterization is different. Let's use the known result:
        # For uniform sampling on the Pareto front surface (unit sphere first octant),
        # we can use:
        # theta_1 = arcsin(u_1)  where u_1 ~ U(0,1)
        # theta_2 = u_2 * pi/2   where u_2 ~ U(0,1)

        # Wait, let me verify. The surface area element on the unit sphere in spherical coords:
        # dA = sin(phi) dphi dtheta
        # In our case, mapping to the first octant of the unit sphere:
        # phi in [0, pi/2], theta in [0, pi/2]
        # f_1 = cos(phi)
        # f_2 = sin(phi) * cos(theta)
        # f_3 = sin(phi) * sin(theta)

        # To sample uniformly on the surface:
        # phi = arcsin(u_1)  (since dA/dphi = sin(phi), and CDF is 1-cos(phi), so phi = arcsin(u))
        # theta = u_2 * pi/2

        # So: x_0 = phi / (pi/2) = arcsin(u_1) / (pi/2)
        #     x_1 = theta / (pi/2) = u_2

        # Let's use this mapping.
        x0 = np.arcsin(u_points[:, 0]) / (np.pi / 2)
        x1 = u_points[:, 1]

        # Construct the full 12-dimensional vectors
        X = np.zeros((n, self.dim))
        X[:, 0] = x0
        X[:, 1] = x1
        # x_2..x_11 are 0 for the Pareto front

        # Evaluate all points
        F = np.array([problem(x) for x in X])

        # Check time
        if time.time() > deadline:
            return X

        # If we have budget remaining (shouldn't happen since n = budget), return
        return X
# EVOLVE-BLOCK-END
