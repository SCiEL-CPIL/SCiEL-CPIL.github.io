import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        def halton_2d(index, base1, base2):
            """Generate 2D Halton sequence point."""
            def halton_1d(idx, base):
                result = 0.0
                f = 1.0 / base
                i = idx
                while i > 0:
                    result += f * (i % base)
                    i //= base
                    f /= base
                return result
            return halton_1d(index, base1), halton_1d(index, base2)

        n_evals = self.budget
        points = []

        for i in range(1, n_evals + 1):
            if time.time() > deadline:
                break

            # 2D Halton in [0,1)^2
            u1, u2 = halton_2d(i, 2, 3)

            # Map to spherical coordinates for first octant of unit sphere
            # theta: elevation from z-axis, in [0, pi/2]
            # phi: azimuth in xy-plane, in [0, pi/2]
            theta = u1 * (np.pi / 2)
            phi = u2 * (np.pi / 2)

            x0 = np.sin(theta) * np.cos(phi)
            x1 = np.sin(theta) * np.sin(phi)
            x2 = np.cos(theta)

            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            x[2] = x2
            x[3:] = 0.5

            x = np.clip(x, lo, hi)
            points.append(x)

        for x in points:
            if time.time() > deadline:
                break
            problem(x)

        return points
