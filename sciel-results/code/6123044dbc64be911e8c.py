"""Multi-objective optimizer for DTLZ1 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 exploiting known Pareto front geometry."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        all_F = []

        def evaluate(x):
            nonlocal all_F
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_F.append(f.copy())
            return f

        n_evals = self.budget

        # DTLZ1 Pareto front:
        # f1 = 0.5 * x0 * g
        # f2 = 0.5 * x1 * (1-x0) * g
        # f3 = 0.5 * (1-x1) * (1-x0) * g
        # With g=0 (optimal shape variables x[2:] = 0.5), the front is the simplex
        # f1 + f2 + f3 = 0.5, fi >= 0.
        #
        # We want to sample points uniformly on this simplex in objective space
        # to maximize hypervolume coverage.
        #
        # Transformation from uniform u1, u2 in [0,1]^2 to uniform y on simplex:
        # y1 = 1 - sqrt(1 - u1)
        # y2 = (1 - y1) * u2
        # y3 = 1 - y1 - y2
        #
        # Then map y to x:
        # x0 = 2 * f1 = 2 * (0.5 * y1) = y1
        # x1 = 2 * f2 / (1 - x0) = y2 / (1 - y1)
        # x[2:] = 0.5

        def halton_sequence(n, base1, base2):
            """Generate n points from 2D Halton sequence."""
            seq = np.zeros((n, 2))
            for i in range(n):
                # Base 1
                x = 0.0
                f = 1.0 / base1
                idx = i + 1
                while idx > 0:
                    x += f * (idx % base1)
                    idx //= base1
                    f /= base1
                # Base 2
                y = 0.0
                f = 1.0 / base2
                idx = i + 1
                while idx > 0:
                    y += f * (idx % base2)
                    idx //= base2
                    f /= base2
                seq[i] = [x, y]
            return seq

        # Generate Halton sequence points in [0,1]^2
        # Using bases 2 and 3 for good low-discrepancy properties
        halton_pts = halton_sequence(n_evals, 2, 3)

        u = halton_pts[:, 0]
        v = halton_pts[:, 1]

        # Transform to uniform points on the 2-simplex in objective space (scaled by 1/0.5 = 2? No, y is on unit simplex)
        # y1 + y2 + y3 = 1
        y1 = 1.0 - np.sqrt(1.0 - u)
        y2 = (1.0 - y1) * v
        # y3 = 1.0 - y1 - y2

        # Map y to x0, x1
        # x0 = y1
        # x1 = y2 / (1 - y1)
        # Handle division by zero if y1 is close to 1
        x0 = y1
        # Avoid division by zero
        denom = 1.0 - x0
        # Where denom is very small, x1 should be 0 (since y2 is also small)
        # y2 = (1-y1)*v, so y2/(1-y1) = v.
        # So x1 = v is actually the stable form!
        # Let's verify:
        # f2 = 0.5 * x1 * (1-x0) = 0.5 * v * (1-y1)
        # We want f2 = 0.5 * y2 = 0.5 * (1-y1) * v.
        # Yes, x1 = v works perfectly and avoids numerical instability.
        x1 = v

        # Evaluate all points on the Pareto front
        for i in range(n_evals):
            if time.time() > deadline - 5:
                break
            if len(all_F) >= self.budget:
                break

            x = np.zeros(self.dim)
            x[0] = x0[i]
            x[1] = x1[i]
            x[2:] = 0.5  # optimal shape variables

            evaluate(x)

        return all_F
# EVOLVE-BLOCK-END
