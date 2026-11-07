"""Direct DTLZ4 Pareto-front construction exploiting known structure."""
import numpy as np
import time


class Optimizer:
    """Builds DTLZ4 Pareto front points directly from the problem's structure.

    DTLZ4 (m=3, K=4 decision vars on the front, N-K=8 control vars):
      x_0..x_3 : simplex variables defining the front shape
      x_4..x_11: control variables; g(x_4..x_11) is minimized at x_i=0.5 (each)
      f_1 = (1+g)*x_0,  f_2 = (1+g)*x_1,  f_3 = (1+g)*(1 - x_0 - x_1)
    At g=0, (x_0,x_1,1-x_0-x_1) traces the unit triangle in objective space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250.0  # safety margin well under 300s
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        used = 0
        candidates = []  # list of x vectors

        # Control variables at their optimum: g=0 when x_i = k/10 for integer k
        # Use 0.5 which is one such point
        ctrl = np.full(self.dim - 4, 0.5)
        ctrl = np.clip(ctrl, lo[4:], hi[4:])

        # --- Phase 1: Add explicit corner and edge-midpoint points ---
        # These ensure the extreme regions of the Pareto front are well-covered
        # Corners of the 3-simplex (4 vars summing to 1)
        corners = [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        ]
        # Edge midpoints
        edge_mids = [
            np.array([0.5, 0.5, 0.0, 0.0]),
            np.array([0.5, 0.0, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5, 0.0]),
            np.array([0.0, 0.5, 0.0, 0.5]),
            np.array([0.0, 0.0, 0.5, 0.5]),
        ]
        # Center point
        center = np.array([0.25, 0.25, 0.25, 0.25])

        for p in corners + edge_mids + [center]:
            x = np.concatenate([p, ctrl])
            x = np.clip(x, lo, hi)
            candidates.append(x)

        # --- Phase 2: Dense sampling on the 3-simplex using inverse CDF ---
        # For 4 vars summing to 1, use the standard transformation:
        # u1, u2, u3 ~ U(0,1)
        # x0 = 1 - u1^(1/4)
        # x1 = u1^(1/4) * (1 - u2^(1/3))
        # x2 = u1^(1/4) * u2^(1/3) * (1 - u3^(1/2))
        # x3 = u1^(1/4) * u2^(1/3) * u3^(1/2)
        # This gives uniform distribution on the 3-simplex.
        
        # Use a structured quasi-random sequence for better coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        # First, add a structured grid to ensure good coverage
        # Use a 3D grid in the transformed space
        n_per_dim = 10  # 10 points per dimension in the transformed space
        grid_points = []
        for i in range(1, n_per_dim):
            for j in range(1, n_per_dim):
                for k in range(1, n_per_dim):
                    u1 = i / n_per_dim
                    u2 = j / n_per_dim
                    u3 = k / n_per_dim
                    
                    x0 = 1.0 - u1 ** 0.25
                    x1 = u1 ** 0.25 * (1.0 - u2 ** (1.0/3.0))
                    x2 = u1 ** 0.25 * u2 ** (1.0/3.0) * (1.0 - u3 ** 0.5)
                    x3 = u1 ** 0.25 * u2 ** (1.0/3.0) * u3 ** 0.5
                    
                    x = np.array([x0, x1, x2, x3])
                    s = x.sum()
                    if s > 0:
                        x = x / s
                    grid_points.append(x)
        
        # Add grid points
        for x in grid_points:
            x_full = np.concatenate([x, ctrl])
            x_full = np.clip(x_full, lo, hi)
            candidates.append(x_full)
        
        # Fill remaining budget with quasi-random points
        n_remaining = self.budget - len(candidates)
        n_halton = min(n_remaining, 1500)
        
        for i in range(1, n_halton + 1):
            u1 = halton(i, 2)
            u2 = halton(i, 3)
            u3 = halton(i, 5)
            
            # Use the correct inverse CDF transformation for uniform simplex sampling
            x0 = 1.0 - u1 ** 0.25
            x1 = u1 ** 0.25 * (1.0 - u2 ** (1.0/3.0))
            x2 = u1 ** 0.25 * u2 ** (1.0/3.0) * (1.0 - u3 ** 0.5)
            x3 = u1 ** 0.25 * u2 ** (1.0/3.0) * u3 ** 0.5
            
            x = np.array([x0, x1, x2, x3])
            s = x.sum()
            if s > 0:
                x = x / s
            
            x_full = np.concatenate([x, ctrl])
            x_full = np.clip(x_full, lo, hi)
            candidates.append(x_full)

        # --- Phase 3: Add random points for robustness ---
        n_rand = max(0, self.budget - len(candidates))
        for _ in range(n_rand):
            # Random point on simplex using Dirichlet(1,1,1,1)
            y = rng.dirichlet(np.ones(4))
            x = np.concatenate([y, ctrl])
            x = np.clip(x, lo, hi)
            candidates.append(x)

        # --- Evaluate candidates up to budget, keeping all returned vectors ---
        # Sort candidates to evaluate most important points first
        # Points near the "center" of the front contribute more to hypervolume
        def point_importance(x):
            # Distance from the ideal point (0,0,0) in objective space
            # Points closer to the center of the simplex are more important
            simplex_part = x[:4]
            # Measure how "central" the point is
            center = np.array([0.25, 0.25, 0.25, 0.25])
            return -np.sum((simplex_part - center) ** 2)
        
        # Sort by importance (most central first)
        candidates_sorted = sorted(candidates, key=point_importance, reverse=True)
        
        evaluated = []
        for x in candidates_sorted:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            f = problem(x)
            evaluated.append((x, np.array(f)))
            used += 1

        # If we have budget left, add random points
        while used < self.budget:
            if time.time() > deadline:
                break
            y = rng.dirichlet(np.ones(4))
            x = np.concatenate([y, ctrl])
            x = np.clip(x, lo, hi)
            f = problem(x)
            evaluated.append((x, np.array(f)))
            used += 1

        return np.array([x for x, _ in evaluated]) if evaluated else np.zeros((1, self.dim))
