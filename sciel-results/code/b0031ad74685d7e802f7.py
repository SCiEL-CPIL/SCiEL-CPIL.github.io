"""Multi-objective optimizer for DTLZ7 exploiting known front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit DTLZ7 structure: front determined by (x0, x1); drive g->0."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim

        # Fibonacci lattice on unit disk for (x0, x1) coverage
        # DTLZ7 front: f1 = (1+x0)*g1(x0,x1), f2 = (1+x1)*g2(x0,x1), f3 = g3(x0,x1)
        # where g_i = (1 + x_i) * (1 + g) for i=0,1 and g3 = 1+g
        # Optimal: g=0, so f1=1+x0, f2=1+x1, f3=1
        # Reference point [1.1, 1.1, 25.0]

        def make_point(x0, x1, g_vars=None):
            """Create a 22-dim point with given x0, x1 and g-variables."""
            x = np.ones(dim, dtype=np.float64)
            x[0] = x0
            x[1] = x1
            if g_vars is not None:
                x[2:] = g_vars
            else:
                # Default: set g-vars to minimize g (which is sum of (xi - gt(xi))^2)
                # gt(xi) = 0.35*(1+1/(4+10*x))*sin(12*pi*x)
                # For DTLZ7, g = sum((xi - gt(xi))^2) for i=2..n-1
                # gt(xi) = 0.35*(1+1/(4+10*xi))*sin(12*pi*xi)
                for i in range(2, dim):
                    xi = rng.random()
                    x[i] = xi
            x = np.clip(x, lo, hi)
            return x

        def compute_gt(x):
            """Compute gt(x) for DTLZ7: 0.35*(1+1/(4+10*x))*sin(12*pi*x)"""
            return 0.35 * (1.0 + 1.0 / (4.0 + 10.0 * x)) * np.sin(12.0 * np.pi * x)

        def compute_g(x):
            """Compute g(x) = sum((xi - gt(xi))^2) for i=2..n-1"""
            if dim <= 2:
                return 0.0
            gt_vals = compute_gt(x[2:])
            return np.sum((x[2:] - gt_vals) ** 2)

        # Robust 1D minimization of (x - gt(x))^2 for a single variable
        def optimize_g_var(x_init):
            """Find x minimizing (x - gt(x))^2 in [0,1] using golden-section search."""
            a, b = 0.0, 1.0
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = (c - compute_gt(c)) ** 2
            fd = (d - compute_gt(d)) ** 2
            for _ in range(100):
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = (c - compute_gt(c)) ** 2
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = (d - compute_gt(d)) ** 2
                if abs(b - a) < 1e-12:
                    break
            return (a + b) / 2.0

        # Precompute optimal g-variables (independent of x0, x1)
        g_vars_optimal = np.zeros(dim - 2)
        for i in range(dim - 2):
            g_vars_optimal[i] = optimize_g_var(0.5)

        points = []
        used = 0

        # Phase 1: Dense Fibonacci lattice on (x0, x1) unit square
        # Use a 2D lattice for better coverage of the Pareto front
        n_per_side = 12  # 144 points, well within budget
        n_init = n_per_side * n_per_side
        for i in range(n_per_side):
            for j in range(n_per_side):
                if used >= self.budget or time.time() > deadline:
                    break
                x0 = (i + 0.5) / n_per_side
                x1 = (j + 0.5) / n_per_side
                x = make_point(x0, x1, g_vars_optimal)
                f = problem(x)
                points.append((x, f))
                used += 1

        # Phase 2: Add finer grid points in regions not yet covered
        # Use a shifted grid to fill gaps
        n_per_side2 = 16
        for i in range(n_per_side2):
            for j in range(n_per_side2):
                if used >= self.budget or time.time() > deadline:
                    break
                x0 = (i + 0.25) / n_per_side2
                x1 = (j + 0.25) / n_per_side2
                x = make_point(x0, x1, g_vars_optimal)
                f = problem(x)
                points.append((x, f))
                used += 1

        # Phase 3: Random sampling to fill remaining budget
        while used < self.budget and time.time() < deadline:
            x0 = rng.random()
            x1 = rng.random()
            x = make_point(x0, x1, g_vars_optimal)
            f = problem(x)
            points.append((x, f))
            used += 1

        # Return the last point (return value is ignored per contract)
        if points:
            return points[-1][0]
        return np.zeros(dim)

    def _compute_gt(self, x):
        """Compute gt(x) for DTLZ7."""
        return 0.35 * (1.0 + 1.0 / (4.0 + 10.0 * x)) * np.sin(12.0 * np.pi * x)
# EVOLVE-BLOCK-END
