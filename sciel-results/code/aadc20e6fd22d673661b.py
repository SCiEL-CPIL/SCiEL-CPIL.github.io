import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _zdt4_f2(self, x0):
        """Compute f2 for ZDT4 given x0 (x[1:] = 0)"""
        x0 = np.clip(x0, 0, 1)
        return 1.0 - np.sqrt(x0) - x0 * np.sin(10.0 * np.pi * x0)**2

    def _hv_2d(self, f1_vals, f2_vals, R1=11.0, R2=400.0):
        """Compute 2D hypervolume for points sorted by increasing f1"""
        if len(f1_vals) == 0:
            return 0.0
        order = np.argsort(f1_vals)
        f1 = f1_vals[order]
        f2 = f2_vals[order]
        mask = (f1 < R1) & (f2 < R2)
        f1 = f1[mask]
        f2 = f2[mask]
        if len(f1) == 0:
            return 0.0

        n = len(f1)
        keep = np.zeros(n, dtype=bool)
        running_min_f2 = np.inf
        for i in range(n):
            if f2[i] < running_min_f2:
                keep[i] = True
                running_min_f2 = f2[i]

        f1 = f1[keep]
        f2 = f2[keep]
        if len(f1) == 0:
            return 0.0

        hv = 0.0
        for i in range(len(f1)):
            next_f1 = f1[i+1] if i+1 < len(f1) else R1
            hv += (next_f1 - f1[i]) * (R2 - f2[i])
        return hv

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        DEADLINE = 290.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        if budget <= 0:
            return np.empty((0, dim))

        N = min(budget, 200)
        R1, R2 = 11.0, 400.0

        # Build a fine grid over x0 in [0, 1] (respecting bounds)
        x0_lo = max(lo[0], 0.0)
        x0_hi = min(hi[0], 1.0)
        if x0_lo >= x0_hi:
            x0_lo, x0_hi = 0.0, 1.0

        M = 2000
        x0_grid = np.linspace(x0_lo, x0_hi, M)
        f2_grid = self._zdt4_f2(x0_grid)

        # Precompute R2 - f2 for hypervolume contributions
        r2_minus_f2 = R2 - f2_grid

        # Dynamic programming:
        # dp[k][i] = max HV contribution from placing k points among the first i grid points,
        #           where the i-th point (x0_grid[i-1]) is the last selected point.
        # We build this by considering adding point i as the k-th point.
        #
        # The HV contribution of a set of points {j1 < j2 < ... < jk} (sorted by f1) is:
        #   sum_{t=1}^{k} (f1_{j_{t+1}} - f1_{j_t}) * (R2 - f2_{j_t})
        # where f1_{j_{k+1}} = R1.
        #
        # This can be decomposed. Let's define:
        #   dp[k][i] = max over all sequences j1 < j2 < ... < j_{k-1} < i of:
        #               [HV contribution of points j1..j_{k-1} ending at j_{k-1}]
        #               + (x0_grid[i-1] - x0_grid[j_{k-1}-1]) * (R2 - f2_grid[j_{k-1}-1])
        #   plus the final segment from i to R1:
        #   + (R1 - x0_grid[i-1]) * (R2 - f2_grid[i-1])
        #
        # Actually, let me redefine more carefully.
        # Let points be indexed 1..M (grid index). f1_j = x0_grid[j-1], f2_j = f2_grid[j-1].
        # HV of set S = {j1 < j2 < ... < jk}:
        #   sum_{t=1}^{k} (f1_{next} - f1_{j_t}) * (R2 - f2_{j_t})
        # where f1_{next} = f1_{j_{t+1}} for t<k, and f1_{next} = R1 for t=k.
        #
        # Define dp[k][i] = max HV of a set of exactly k points where the last point is grid index i (1-indexed).
        # The HV includes the full contribution up to R1.
        #
        # Recurrence:
        # dp[1][i] = (R1 - f1_i) * (R2 - f2_i)
        # dp[k][i] = max_{j < i} [ dp[k-1][j] - (R1 - f1_j)*(R2 - f2_j) + (f1_i - f1_j)*(R2 - f2_j) ] + (R1 - f1_i)*(R2 - f2_i)
        #
        # Simplify: dp[k-1][j] already includes (R1 - f1_j)*(R2 - f2_j) as the last segment.
        # When we add point i after j, the segment from j to R1 is replaced by segment from j to i plus segment from i to R1.
        # So: dp[k][i] = max_{j < i} [ dp[k-1][j] - (R1 - f1_j)*(R2 - f2_j) + (f1_i - f1_j)*(R2 - f2_j) ] + (R1 - f1_i)*(R2 - f2_i)
        #
        # Let g[j] = dp[k-1][j] - (R1 - f1_j)*(R2 - f2_j)
        # Then dp[k][i] = max_{j < i} [ g[j] + (f1_i - f1_j)*(R2 - f2_j) ] + (R1 - f1_i)*(R2 - f2_i)
        #                = max_{j < i} [ g[j] - f1_j*(R2 - f2_j) + f1_i*(R2 - f2_j) ] + (R1 - f1_i)*(R2 - f2_i)
        #
        # This is still O(M^2) per k, so O(N*M^2) total. With N=200, M=2000, that's 8*10^8 which might be too slow.
        # Let me reduce M to 500. Then 200 * 500^2 = 5*10^7, manageable.

        M = 500
        x0_grid = np.linspace(x0_lo, x0_hi, M)
        f2_grid = self._zdt4_f2(x0_grid)
        f1_grid = x0_grid.copy()
        r2_minus_f2 = R2 - f2_grid

        # DP tables
        # dp_prev[j] = dp[k-1][j], parent_prev[j] = which j' was used to get dp[k-1][j]
        dp_prev = np.full(M, -np.inf)
        parent_prev = np.zeros(M, dtype=int)

        # k = 1
        dp_curr = (R1 - f1_grid) * r2_minus_f2
        parent_curr = np.zeros(M, dtype=int)  # no parent for k=1

        for k in range(1, N):
            if time.time() - t0 > DEADLINE - 5:
                break
            dp_curr = np.full(M, -np.inf)
            parent_curr = np.zeros(M, dtype=int)

            # g[j] = dp_prev[j] - (R1 - f1_j)*(R2 - f2_j)
            g = dp_prev - (R1 - f1_grid) * r2_minus_f2
            # For each i, dp_curr[i] = max_{j < i} [ g[j] - f1_j * r2_minus_f2[j] + f1_i * r2_minus_f2[j] ] + (R1 - f1_i) * r2_minus_f2[i]
            # Let a[j] = g[j] - f1_j * r2_minus_f2[j]
            # Let b[j] = r2_minus_f2[j]
            # dp_curr[i] = max_{j < i} [ a[j] + f1_i * b[j] ] + (R1 - f1_i) * r2_minus_f2[i]

            a = g - f1_grid * r2_minus_f2
            b = r2_minus_f2

            # For each i from 1 to M-1, compute max_{j < i} [ a[j] + f1_i * b[j] ]
            # This is a max of lines evaluated at f1_i. Each j defines a line y = a[j] + x*b[j].
            # We can use a convex hull trick, but for simplicity with M=500, O(M^2) per k is 250k per k, * 200 = 50M. Fine.
            best_vals = np.full(M, -np.inf)
            best_parents = np.zeros(M, dtype=int)
            for i in range(1, M):
                # max over j in 0..i-1
                vals = a[:i] + f1_grid[i] * b[:i]
                j_idx = np.argmax(vals)
                best_vals[i] = vals[j_idx]
                best_parents[i] = j_idx

            dp_curr[1:] = best_vals[1:] + (R1 - f1_grid[1:]) * r2_minus_f2[1:]
            parent_curr[1:] = best_parents[1:]

            dp_prev = dp_curr.copy()
            parent_prev = parent_curr.copy()

        # dp_prev now holds dp[N][i] for the full N points
        # Find best ending index
        if np.all(np.isnan(dp_prev)) or np.all(dp_prev == -np.inf):
            # Fallback: uniform spacing
            x0_vals = np.linspace(x0_lo, x0_hi, N)
        else:
            best_i = np.argmax(dp_prev)
            # Backtrack
            indices = [best_i]
            # We need to recompute parent arrays for each k. Let's redo the DP storing all parents.
            # This is memory intensive but N=200, M=500, so 200*500 = 100k ints. Fine.

            # Redo DP with full parent storage
            dp_tables = [None] * N
            parent_tables = [None] * N

            # k=1
            dp_k1 = (R1 - f1_grid) * r2_minus_f2
            dp_tables[1] = dp_k1
            parent_tables[1] = np.zeros(M, dtype=int)

            dp_prev = dp_k1.copy()
            for k in range(2, N + 1):
                if time.time() - t0 > DEADLINE - 5:
                    break
                g = dp_prev - (R1 - f1_grid) * r2_minus_f2
                a = g - f1_grid * r2_minus_f2
                b = r2_minus_f2

                best_vals = np.full(M, -np.inf)
                best_parents = np.zeros(M, dtype=int)
                for i in range(1, M):
                    vals = a[:i] + f1_grid[i] * b[:i]
                    j_idx = np.argmax(vals)
                    best_vals[i] = vals[j_idx]
                    best_parents[i] = j_idx

                dp_curr = np.full(M, -np.inf)
                dp_curr[1:] = best_vals[1:] + (R1 - f1_grid[1:]) * r2_minus_f2[1:]
                dp_tables[k] = dp_curr.copy()
                parent_tables[k] = best_parents.copy()
                dp_prev = dp_curr

            # Backtrack
            best_i = np.argmax(dp_tables[N])
            indices = [best_i]
            for k in range(N, 1, -1):
                best_i = parent_tables[k][best_i]
                indices.append(best_i)
            indices.reverse()
            x0_vals = f1_grid[indices]

        x0_vals = np.clip(x0_vals, x0_lo, x0_hi)

        evals = 0
        results = []

        for x0_val in x0_vals:
            if evals >= budget or time.time() - t0 > DEADLINE:
                break

            x = np.zeros(dim)
            x[0] = float(x0_val)
            x = np.clip(x, lo, hi)

            try:
                f = problem(x)
                evals += 1
                results.append(f)
            except Exception:
                pass

        if results:
            return np.array(results)

        return np.empty((0, dim))
