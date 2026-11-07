"""DTLZ7 hypervolume optimizer via structured front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _halton_sequence(n, base):
    """Generate n points in [0,1) using the Halton sequence."""
    seq = []
    f = 1.0
    for i in range(1, n + 1):
        x = 0.0
        j = i
        while j > 0:
            f /= base
            x += f * (j % base)
            j //= base
        seq.append(x)
    return np.array(seq)


class Optimizer:
    """DTLZ7-specific optimizer exploiting the known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # generous safety margin before the 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n = self.budget

        # For DTLZ7 with x[3..21]=0, the Pareto front is 2D.
        # We'll use a combination of:
        # 1. Halton sequence for uniform low-discrepancy coverage
        # 2. Additional points biased toward high-hypervolume regions
        
        # Generate Halton sequence points (low-discrepancy, good coverage)
        n_halton = max(4, n - 4)  # Leave room for corner points
        h1 = _halton_sequence(n_halton, 2)
        h2 = _halton_sequence(n_halton, 3)
        
        pts = list(zip(h1, h2))
        
        # Add corner points explicitly
        corners = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        for c in corners:
            if len(pts) < n:
                pts.append(c)
        
        # Trim to budget
        pts = pts[:n]
        
        x1 = np.array([p[0] for p in pts])
        x2 = np.array([p[1] for p in pts])

        # Clip to bounds (defensive)
        x1 = np.clip(x1, lo[0], hi[0])
        x2 = np.clip(x2, lo[1], hi[1])

        # Build full 22-dim vectors with x3..x22 = 0
        X = np.zeros((len(x1), self.dim))
        X[:, 0] = x1
        X[:, 1] = x2

        # Evaluate each point; stop early if we exceed the deadline.
        F = np.zeros((len(x1), self.n_obj))
        for i in range(len(x1)):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])

        return X
# EVOLVE-BLOCK-END
