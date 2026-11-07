"""DTLZ1 optimizer: exploit known Pareto front with hypervolume-optimal density sampling."""
import numpy as np
import time


class Optimizer:
    """
    Hypervolume-optimal density sampling for DTLZ1:
    1. Calibrate K (sum of objectives on the Pareto front) empirically.
    2. Sample the simplex f1+f2+f3=K using a density proportional to
       1/((R-f1)(R-f2)(R-f3)) for uniform hypervolume coverage.
    3. Use inverse CDF transformation with Halton low-discrepancy sequence.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 280.0  # safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        ref = np.array([400.0, 400.0, 400.0])

        used = 0
        results = []

        def eval_point(x):
            nonlocal used
            if used >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.array(problem(x_clipped), dtype=float)
            results.append(f)
            used += 1
            return f

        # Step 1: Calibrate K by probing the problem
        # DTLZ1: g(x_tail) = 1 + 10*|M|, where M = number of tail vars != 0.5
        # On Pareto front, g = g_min, so f1+f2+f3 = (1-g_min)/2
        # For dim=7, n_obj=3, tail has 4 variables.
        # If all tail vars = 0.5, M=0, g_min=1, K=(1-1)/2=0 (degenerate)
        # So we need to find the actual g_min by probing.

        tail_candidates = [
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.5],
            [0.1, 0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9, 0.9],
            [0.3, 0.3, 0.3, 0.3],
            [0.7, 0.7, 0.7, 0.7],
        ]

        min_sum = float('inf')
        best_tail = None
        best_x = None

        for tail in tail_candidates:
            if used >= budget or time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = 0.5
            x[1] = 0.5
            x[2] = 0.5
            for i, t in enumerate(tail):
                x[3 + i] = t
            f = eval_point(x)
            if f is not None:
                s = float(np.sum(f))
                if s < min_sum:
                    min_sum = s
                    best_tail = tail[:]
                    best_x = x.copy()

        if min_sum == float('inf') or used >= budget or time.time() > deadline:
            return results

        K = max(min_sum, 1e-10)

        # Step 2: Map simplex points to decision space
        def simplex_to_decision(f1, f2, f3, K_val, tail):
            x = np.zeros(dim)
            for i, t in enumerate(tail):
                x[3 + i] = t
            if K_val < 1e-12:
                x[0] = 0.5
                x[1] = 0.5
                x[2] = 0.5
                return np.clip(x, lo, hi)
            x[0] = f1 / K_val
            x[1] = 1.0 - f3 / K_val
            x[2] = 0.5
            return np.clip(x, lo, hi)

        remaining = budget - used
        if remaining <= 0 or time.time() > deadline:
            return results

        # Step 3: Hypervolume-optimal density sampling
        # For 3 objectives with reference point R, the marginal hypervolume
        # contribution of a point at (f1, f2, f3) on the simplex is
        # proportional to 1/((R-f1)(R-f2)(R-f3)) when K << R.
        # We sample with density proportional to this weight.

        # Use Halton sequence and inverse CDF transformation
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        # Build a CDF for the optimal density on the simplex
        # The simplex is f1+f2+f3=K, fi>=0
        # Parametrize: f1 = K*u, f2 = K*(1-u)*v, f3 = K*(1-u)*(1-v)
        # where u,v in [0,1]
        # Weight w(u,v) = 1/((R-K*u)*(R-K*(1-u)*v)*(R-K*(1-u)*(1-v)))
        # Since K << R, w ≈ 1/R^3 (nearly uniform)
        # But for better accuracy, use the actual weight.

        # Discretize the (u,v) unit square into a grid to build CDF
        n_grid = 100
        u_vals = np.linspace(0, 1, n_grid + 1)
        v_vals = np.linspace(0, 1, n_grid + 1)
        du = u_vals[1] - u_vals[0]
        dv = v_vals[1] - v_vals[0]

        # Compute weights on grid
        weights = np.zeros((n_grid, n_grid))
        for i in range(n_grid):
            u_mid = (u_vals[i] + u_vals[i+1]) / 2
            for j in range(n_grid):
                v_mid = (v_vals[j] + v_vals[j+1]) / 2
                f1 = K * u_mid
                f2 = K * (1 - u_mid) * v_mid
                f3 = K * (1 - u_mid) * (1 - v_mid)
                w1 = max(ref[0] - f1, 1e-6)
                w2 = max(ref[1] - f2, 1e-6)
                w3 = max(ref[2] - f3, 1e-6)
                weights[i, j] = 1.0 / (w1 * w2 * w3)

        # Build 2D CDF
        cdf = np.cumsum(weights)
        total_weight = cdf[-1, -1]
        cdf_2d = cdf / total_weight

        # Sample using inverse CDF with Halton
        # For each Halton point (h1, h2), find (u, v) such that CDF(u,v) = (h1, h2)
        # This is an approximation; use bilinear interpolation on the grid

        def sample_optimal_density(h1, h2, cdf_2d, n_grid):
            """Invert 2D CDF using bisection + linear search."""
            # For h1: find u such that CDF(u, 1) ≈ h1
            # CDF(u, 1) = sum of weights for columns 0..col(u)
            col_sums = cdf_2d[:, -1]  # cumulative in u direction at v=1
            # Find i such that col_sums[i] <= h1 < col_sums[i+1]
            i_u = np.searchsorted(col_sums, h1)
            i_u = min(max(i_u, 0), n_grid - 1)
            # Interpolate u
            if i_u < n_grid - 1 and col_sums[i_u + 1] > col_sums[i_u]:
                frac = (h1 - col_sums[i_u]) / (col_sums[i_u + 1] - col_sums[i_u])
                u = (i_u + frac) / n_grid
            else:
                u = i_u / n_grid

            # For h2: given u, find v such that CDF(u, v) / CDF(u, 1) ≈ h2
            # CDF(u, v) for fixed u is the row sum up to v
            # Approximate: use the row at i_u
            row = cdf_2d[i_u, :]
            # Normalize by the row total at v=1
            row_total = row[-1] if row[-1] > 0 else 1.0
            target = h2 * row_total
            j_v = np.searchsorted(row, target)
            j_v = min(max(j_v, 0), n_grid - 1)
            if j_v < n_grid - 1 and row[j_v + 1] > row[j_v]:
                frac2 = (target - row[j_v]) / (row[j_v + 1] - row[j_v])
                v = (j_v + frac2) / n_grid
            else:
                v = j_v / n_grid

            return u, v

        # Evaluate points
        # Add strategic points first: vertices, edge midpoints, center
        eps = 1e-8
        strategic = [
            (K - eps, eps, eps),
            (eps, K - eps, eps),
            (eps, eps, K - eps),
            (K/2, K/2, eps),
            (K/2, eps, K/2),
            (eps, K/2, K/2),
            (K/3, K/3, K/3),
        ]

        for f1, f2, f3 in strategic:
            if used >= budget or time.time() > deadline:
                break
            x = simplex_to_decision(f1, f2, f3, K, best_tail)
            eval_point(x)

        remaining = budget - used
        if remaining <= 0 or time.time() > deadline:
            return results

        # Main sampling loop with optimal density
        offset = len(strategic)
        for idx in range(remaining):
            if used >= budget or time.time() > deadline:
                break
            h1 = halton(idx + offset + 1, 2)
            h2 = halton(idx + offset + 1, 3)
            u, v = sample_optimal_density(h1, h2, cdf_2d, n_grid)
            f1 = K * u
            f2 = K * (1 - u) * v
            f3 = K * (1 - u) * (1 - v)
            x = simplex_to_decision(f1, f2, f3, K, best_tail)
            eval_point(x)

        return results
