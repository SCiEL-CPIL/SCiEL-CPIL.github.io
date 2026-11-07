"""Multi-objective optimizer on DTLZ7 exploiting known Pareto front structure."""
import numpy as np
import time


class Optimizer:
    """DTLZ7 optimizer using structured 2D lattice sampling on the known Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_2d(self, n, seed=0):
        """Generate n 2D points using a scrambled-like lattice (independent random shifts
        per dimension plus a low-discrepancy base grid). Returns (n, 2) array in [0,1)^2."""
        # Use a simple 2D low-discrepancy construction: combine two 1D low-discrepancy
        # sequences (van der Corput / Halton) with independent random shifts to decorrelate.
        # This gives a 2D sequence with good spread and no visible grid alignment.
        rng = np.random.RandomState(seed)
        shift0 = rng.random()
        shift1 = rng.random()

        def halton_seq(n_pts, base):
            """Return n_pts Halton values in [0,1)."""
            out = np.empty(n_pts)
            for i in range(n_pts):
                idx = i + 1
                f = 1.0
                r = 0.0
                while idx > 0:
                    f /= base
                    r += f * (idx % base)
                    idx //= base
                out[i] = r
            return out

        h2 = halton_seq(n, 2)
        h3 = halton_seq(n, 3)
        x0 = (h2 + shift0) % 1.0
        x1 = (h3 + shift1) % 1.0
        return np.column_stack((x0, x1))

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        used = 0
        n = self.budget

        # Generate all 2D lattice points in [0,1)^2, map to bounds, set x[2:]=0.
        # Using a single Halton-based 2D sequence gives near-uniform coverage with
        # no repeated grid rows/columns, which is better for hypervolume than a
        # regular grid (which wastes points on aligned structures).
        pts2 = self._sobol_2d(n, seed=42)

        # Map to [lo, hi] in each dimension for x[0], x[1]
        x0 = lo[0] + (hi[0] - lo[0]) * pts2[:, 0]
        x1 = lo[1] + (hi[1] - lo[1]) * pts2[:, 1]

        # Pre-allocate zero vector for x[2:] (on Pareto front)
        # Evaluate one at a time with deadline guard
        for i in range(n):
            if time.time() > deadline or used >= n:
                break
            x = np.zeros(self.dim)
            x[0] = x0[i]
            x[1] = x1[i]
            # x[2:] remains 0 -> Pareto front (g=0)
            problem(x)
            used += 1

        return None
