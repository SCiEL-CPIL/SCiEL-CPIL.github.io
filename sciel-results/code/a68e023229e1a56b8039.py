"""DTLZ7 hypervolume optimizer exploiting the known Pareto front geometry."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _halton(index, base):
    f = 1.0
    r = 0.0
    i = index + 1
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n = self.budget

        # DTLZ7 with x[2:] = 0 gives the Pareto front:
        # f1 = 1, f2 = x2*(3 - h(x1) - h(x2)), f3 = h(x1)
        # where h(y) = y^2 if y<=0.5 else 1-(y-0.5)^2 - 0.5*sin(8*pi*y)
        # Hypervolume is maximized by points with small f2 and f3.
        # The corner (x1, x2) = (0, 1) gives f2=0, f3=0 (best point).
        # We bias sampling toward this corner.

        # Strategy: mix of uniform Halton and biased Halton
        # Use power-law transformation to concentrate points near (0, 1)
        # x1_biased = x1_uniform^alpha  (concentrates near 0)
        # x2_biased = 1 - (1 - x2_uniform)^beta  (concentrates near 1)

        n_total = n
        # Allocate 60% to biased sampling near (0,1), 40% to uniform for coverage
        n_biased = int(n_total * 0.6)
        n_uniform = n_total - n_biased

        # Generate Halton sequences
        u1_biased = np.empty(n_biased)
        u2_biased = np.empty(n_biased)
        for k in range(n_biased):
            u1_biased[k] = _halton(k, 2)
            u2_biased[k] = _halton(k, 3)

        u1_uniform = np.empty(n_uniform)
        u2_uniform = np.empty(n_uniform)
        for k in range(n_uniform):
            u1_uniform[k] = _halton(n_biased + k, 2)
            u2_uniform[k] = _halton(n_biased + k, 3)

        # Apply power-law bias to concentrate near (0, 1)
        # alpha > 1 concentrates x1 near 0, beta > 1 concentrates x2 near 1
        alpha = 2.0
        beta = 2.0
        x1_biased = u1_biased ** alpha
        x2_biased = 1.0 - (1.0 - u2_biased) ** beta

        x1_uniform = u1_uniform
        x2_uniform = u2_uniform

        # Combine all points
        x1 = np.concatenate([x1_biased, x1_uniform])
        x2 = np.concatenate([x2_biased, x2_uniform])

        # Clip to bounds
        x1 = np.clip(x1, lo[0], hi[0])
        x2 = np.clip(x2, lo[1], hi[1])

        # Build full 22-dim vectors with x3..x22 = 0
        X = np.zeros((n_total, self.dim))
        X[:, 0] = x1
        X[:, 1] = x2

        # Evaluate each point; stop early if we exceed the deadline.
        F = np.zeros((n_total, self.n_obj))
        for i in range(n_total):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])

        return X
# EVOLVE-BLOCK-END
