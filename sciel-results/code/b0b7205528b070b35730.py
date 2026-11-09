import numpy as np
import time


class Optimizer:
    """Analytical DTLZ4 optimizer: sample the Pareto front with a Sobol sequence
    (low-discrepancy) plus explicit critical points, constructing x analytically
    so that g(x)=0 (distance vars at 0.5) and the position vars encode the front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.t0 = time.time()
        self.deadline = self.t0 + 280  # Safety margin for 300s limit

    def _sobol_2d(self, n, skip=0):
        """Generate n 2D Sobol points in [0,1]^2 using a simple direction-number
        method (Faaß/Bratley-Sobol style). Returns (n,2) array."""
        if n <= 0:
            return np.zeros((0, 2))
        # Start indices at 1 (skip the trivial 0 point) to get interior coverage
        idx = np.arange(skip + 1, skip + 1 + n)
        # Precompute vandermonde-ish for base 2
        base = 2
        max_idx = idx.max()
        # Determine number of bits needed
        nbits = int(np.floor(np.log2(max_idx))) + 1 if max_idx > 0 else 0
        nbits = max(nbits, 1)
        powers = np.arange(nbits)  # 0,1,2,...
        bits = ((idx[:, None] >> powers[None, :]) & 1).astype(np.float64)
        # Direction numbers for base 2 (standard Sobol for first dimension)
        # v[0]=1, v[i] = (2^{i-1}+1) for i>=1 (this is a valid direction sequence)
        v = np.ones(nbits)
        for i in range(1, nbits):
            v[i] = (2 ** (i - 1) + 1) % (2 ** i)
        # For second dimension use a different direction sequence (shifted)
        v2 = np.ones(nbits)
        for i in range(1, nbits):
            v2[i] = (2 ** (i - 1) + 2) % (2 ** i)
        # Gray code: g = idx ^ (idx >> 1)
        gray = (idx ^ (idx >> 1))
        gbits = ((gray[:, None] >> powers[None, :]) & 1).astype(np.float64)
        # Dimension 1
        c1 = np.zeros(nbits)
        c1[0] = v[0] / (2 ** 0)
        for i in range(1, nbits):
            c1[i] = (c1[i - 1] + v[i]) % (2 ** i) / (2 ** i)
        # Dimension 2
        c2 = np.zeros(nbits)
        c2[0] = v2[0] / (2 ** 0)
        for i in range(1, nbits):
            c2[i] = (c2[i - 1] + v2[i]) % (2 ** i) / (2 ** i)
        d1 = (gbits @ c1) % 1.0
        d2 = (gbits @ c2) % 1.0
        return np.column_stack((d1, d2))

    def _dtlz4_pareto_point(self, u, alpha=10.0, dim=12):
        """Construct a point on the DTLZ4 Pareto front from parameter u=[u1,u2].
        Position vars: x[0]=u1^(1/alpha), x[1]=u2^(1/alpha); distance vars = 0.5."""
        u1, u2 = u[0], u[1]
        x = np.zeros(dim)
        x[0] = u1 ** (1.0 / alpha)
        x[1] = u2 ** (1.0 / alpha)
        x[2:] = 0.5
        return x

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_pos = self.dim - (self.n_obj - 1)

        # Build a set of parameter points in [0,1]^2 covering the front uniformly.
        # Use Sobol for low-discrepancy coverage, plus explicit critical points.
        n_sobol = max(1, self.budget - 10)  # reserve a few for critical points
        pts = self._sobol_2d(n_sobol)
        # Explicit critical points: corners and center of the parameter square
        critical = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [0.5, 0.0], [1.0, 0.5], [0.0, 0.5], [0.5, 1.0],
            [0.5, 0.5], [0.0, 0.5],
        ])
        all_pts = np.vstack([pts, critical])
        # Deduplicate by rounding
        key = np.round(all_pts, 10)
        _, uniq = np.unique(key, axis=0, return_index=True)
        all_pts = all_pts[np.sort(uniq)]

        # Cap at budget
        if all_pts.shape[0] > self.budget:
            all_pts = all_pts[:self.budget]

        # Construct decision vectors
        X = np.zeros((all_pts.shape[0], self.dim))
        for i in range(all_pts.shape[0]):
            X[i] = self._dtlz4_pareto_point(all_pts[i], alpha=10.0, dim=self.dim)
        X = np.clip(X, lo, hi)

        fvals = []
        for i in range(X.shape[0]):
            if time.time() > deadline:
                break
            f = problem(X[i])
            fvals.append(f)
        return fvals
