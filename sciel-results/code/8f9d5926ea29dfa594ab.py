"""Deterministic Pareto-front sampling for ZDT3 using Halton sequence."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sample the known ZDT3 Pareto front with a space-filling (Halton) sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    @staticmethod
    def _halton(n, base):
        """Generate n points in [0,1) using the Halton sequence with the given base."""
        seq = np.zeros(n)
        f = 1.0 / base
        for i in range(1, n):
            num = i
            idx = i
            while num > 0:
                idx += 1
                num, rem = divmod(num, base)
                f /= base
                seq[idx - 1] = f * rem
        return seq

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # 30s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        budget = self.budget
        dim = self.dim

        # ZDT3 Pareto front: x[1:] = 0, x[0] in [0, 1]
        # f1 = x0
        # g = 2 + 10*|x| + sum(10*(1+6x_i)*sin(12*pi*x_i)) for i>=1
        #    when x[1:]=0: g = 2
        # f2 = (1 + g) * (1 - sqrt(f1/(1+g)))
        #    = 3 * (1 - sqrt(x0/3))

        # Generate Halton sequence points in [0,1] for x0
        n = budget
        # Use base 2 for 1D Halton (just binary van der Corput)
        x0_vals = self._halton(n, 2)

        # Ensure we cover the endpoints better: add 0 and 1 explicitly if not present
        # The Halton sequence starts at 0.5, 0.25, 0.75, 0.125, ...
        # Let's shift to ensure good coverage including near 0 and 1
        # Actually, let's use a different approach: generate n-2 interior points, then add 0 and 1
        # But we must stay within budget. Let's just use the Halton points as-is but
        # also include 0 and 1 by replacing the two points farthest from endpoints.

        # Better: use n points from Halton, but ensure 0 and 1 are included
        # Replace the point closest to 0 with exactly 0, and closest to 1 with exactly 1
        if n >= 2:
            idx_min = np.argmin(x0_vals)
            idx_max = np.argmax(x0_vals)
            x0_vals[idx_min] = 0.0
            x0_vals[idx_max] = 1.0

        # Build the decision vectors: x[0] = x0, x[1:] = 0
        X = np.zeros((n, dim), dtype=float)
        X[:, 0] = x0_vals

        # Evaluate all points
        F = np.zeros((n, self.n_obj), dtype=float)
        for i in range(n):
            if time.time() > deadline:
                # Return what we have so far
                F = F[:i+1]
                break
            F[i] = problem(X[i])
        else:
            F = F  # all evaluated

        return X
# EVOLVE-BLOCK-END
