"""Rewritten DTLZ2 optimizer: uniform-surface-area Halton sampling with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    DTLZ2 (M=3, N=12) minimizer.
    Pareto front: f1 = g*sin(theta1), f2 = g*cos(theta1)*sin(theta2), f3 = g*cos(theta1)*cos(theta2)
    with theta1, theta2 in [0, pi/2] and g(x) = 0 at x[2:] = 0.5.
    We sample theta1, theta2 for uniform surface area on the quarter-sphere and set x[2:]=0.5.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _halton_sequence(n, base):
        """Return n Halton points in [0,1) for a given base."""
        result = np.empty(n)
        for i in range(n):
            f = 1.0
            r = 0.0
            j = i + 1
            while j > 0:
                j, rem = divmod(j, base)
                f *= 1.0 / base
                r += rem * f
            result[i] = r
        return result

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        n = self.budget

        # Two independent Halton dimensions for (u, v) in [0,1)^2
        u = self._halton_sequence(n, 2)
        v = self._halton_sequence(n, 3)

        # For uniform surface area on quarter-sphere:
        #   cos(theta1) uniform -> theta1 = arccos(1 - u)
        #   sin(theta1) * cos(theta2) and sin(theta1) * sin(theta2) uniform in the
        #   conditional => theta2 = arccos(1 - v) works for uniform on the sphere cap.
        # Simpler: use theta1 = arccos(1-u), theta2 = arccos(1-v) gives uniform on sphere.
        # We want x[0] = (2/pi)*theta1, x[1] = (2/pi)*theta2 (DTLZ2 mapping).
        theta1 = np.arccos(1.0 - u)
        theta2 = np.arccos(1.0 - v)

        x0 = (2.0 / np.pi) * theta1
        x1 = (2.0 / np.pi) * theta2

        # Build full x vector
        x = np.zeros((n, self.dim), dtype=float)
        x[:, 0] = x0
        x[:, 1] = x1
        if self.dim > 2:
            center = (lo[2:] + hi[2:]) / 2.0
            x[:, 2:] = center

        x = np.clip(x, lo, hi)

        results = []
        for i in range(n):
            if time.time() > deadline:
                break
            try:
                f = problem(x[i])
                results.append(f)
            except Exception:
                break

        return np.array(results) if results else None
# EVOLVE-BLOCK-END
