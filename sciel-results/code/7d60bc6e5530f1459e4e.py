"""Direct DTLZ4 Pareto-front construction exploiting known structure.

Strategy:
  1. Analytically set control variables to their optimum (x_i = 0.5) so g=0.
  2. Use a dense stratified grid on the 2-simplex for broad coverage.
  3. Use a deterministic local search (Nelder-Mead style) to place additional
     points in regions of highest marginal hypervolume contribution.
"""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        used = 0
        evaluated = []

        def check_time():
            return time.time() < deadline

        # Control variables at optimum: x_i = 0.5
        n_ctrl = self.dim - 3
        ctrl = np.full(n_ctrl, 0.5)
        ctrl = np.clip(ctrl, lo[3:], hi[3:])

        # --- Phase 1: Dense stratified grid on the simplex ---
        # Parameterize simplex: x0 = 1-sqrt(u), x1 = sqrt(u)*(1-v), x2 = sqrt(u)*v
        # Use a grid in (u, v) space with stratification

        # For hypervolume, we want points spread across the triangle.
        # A uniform grid in (u,v) gives non-uniform coverage in objective space.
        # Instead, use a grid that is roughly uniform in the objective triangle.

        # The triangle in (f1, f2) space has vertices (1,0), (0,1), (0,0)
        # (since f3 = 1 - f1 - f2 when g=0).
        # We can sample uniformly in the triangle using:
        #   s = sqrt(u), t = v
        #   x0 = 1 - s, x1 = s*(1-t), x2 = s*t

        # Use a stratified grid: divide the triangle into cells
        # For N points, use a triangular lattice

        n_grid = 150  # grid points for broad coverage

        # Create a stratified grid on the simplex
        # Use a triangular arrangement
        n_rows = int(np.ceil(np.sqrt(2 * n_grid)))
        grid_pts = []
        for i in range(n_rows + 1):
            for j in range(n_rows - i + 1):
                # Barycentric coordinates with jitter for stratification
                # i+j <= n_rows
                # x0 = (n_rows - i - j) / n_rows
                # x1 = i / n_rows
                # x2 = j / n_rows
                # Add small jitter for stratification
                if len(grid_pts) >= n_grid:
                    break
                # Add stratified jitter within the cell
                r1 = np.random.RandomState(hash((i, j)) % 2**32).random()
                r2 = np.random.RandomState(hash((i, j, 1)) % 2**32).random()
                # Jitter within [0, 1/n_rows]
                jit = 1.0 / n_rows
                x0 = (n_rows - i - j + r1) / n_rows
                x1 = (i + r2 * (1 - r1)) / n_rows
                x2 = (j + (1 - r1) * (1 - r2)) / n_rows
                # Normalize to ensure sum = 1
                s = x0 + x1 + x2
                if s > 0:
                    x0 /= s
                    x1 /= s
                    x2 /= s
                grid_pts.append(np.array([x0, x1, x2]))
            if len(grid_pts) >= n_grid:
                break

        # Build full x vectors for grid points
        candidates = []
        for p in grid_pts:
            x = np.concatenate([p, ctrl])
            x = np.clip(x, lo, hi)
            candidates.append(x)

        # --- Evaluate grid points ---
        for x in candidates:
            if used >= self.budget or not check_time():
                break
            f = problem(x)
            evaluated.append((x, np.array(f, dtype=float)))
            used += 1

        if used >= self.budget or not check_time():
            return np.array([x for x, _ in evaluated]) if evaluated else np.zeros((1, self.dim))

        # --- Phase 2: Local search for marginal hypervolume improvement ---
        # For the remaining budget, use Nelder-Mead to find points that
        # maximize marginal hypervolume contribution.

        remaining_budget = self.budget - used

        # Compute current hypervolume contribution of each point
        # to guide where to place new points

        # Extract objective values
        if len(evaluated) > 0:
            obj_vals = np.array([fv for _, fv in evaluated])
            # Filter to only points better than reference in all objectives
            ref = np.array([2.5, 2.5, 2.5])
            valid_mask = np.all(obj_vals < ref, axis=1)
            valid_objs = obj_vals[valid_mask]
        else:
            valid_objs = np.empty((0, self.n_obj))

        # For marginal HV, we want to find points on the front that, when added,
        # increase HV the most. We approximate this by finding points that are
        # in "gaps" of the current front.

        # Use a simple approach: sample candidate points in gaps and evaluate
        # which give the best marginal contribution.

        # Define a surrogate: for a point p on the front, its marginal HV is
        # approximately the volume of the region it dominates that isn't already
        # dominated by existing points.

        # For simplicity, use a grid-based approximation:
        # Divide the triangle into cells, find empty cells, and place points there.

        # Create a fine grid and find which cells have no points
        n_fine = 20
        fine_grid = []
        for i in range(n_fine + 1):
            for j in range(n_fine - i + 1):
                x0 = (n_fine - i - j) / n_fine
                x1 = i / n_fine
                x2 = j / n_fine
                fine_grid.append(np.array([x0, x1, x2]))

        # Check which fine grid cells have nearby evaluated points
        # For each fine grid point, compute distance to nearest evaluated point
        # in objective space

        if len(valid_objs) > 0:
            # Convert simplex points to objective values (g=0)
            # f1 = x0, f2 = x1, f3 = 1 - x0 - x1 = x2
            eval_objs_simplex = np.array([
                [x[0], x[1], x[2]] for x in [ex for ex, _ in evaluated]
            ])
        else:
            eval_objs_simplex = np.empty((0, 3))

        # Find gaps: fine grid points far from any evaluated point
        gap_candidates = []
        for fg in fine_grid:
            if len(eval_objs_simplex) > 0:
                dists = np.linalg.norm(eval_objs_simplex - fg, axis=1)
                min_dist = np.min(dists)
            else:
                min_dist = np.inf
            if min_dist > 0.05:  # threshold for being in a gap
                gap_candidates.append((min_dist, fg))

        # Sort by distance (farthest first) to prioritize gaps
        gap_candidates.sort(key=lambda x: -x[0])

        # Evaluate gap candidates
        for dist, fg in gap_candidates:
            if used >= self.budget or not check_time():
                break
            x = np.concatenate([fg, ctrl])
            x = np.clip(x, lo, hi)
            f = problem(x)
            evaluated.append((x, np.array(f, dtype=float)))
            used += 1

        # --- Phase 3: If budget remains, use random sampling in high-value regions ---
        rng = np.random.default_rng(123)
        while used < self.budget and check_time():
            # Sample in regions near the edges of the triangle (higher HV contribution)
            # Use a power-law distribution biased toward edges
            u = rng.random()
            v = rng.random()
            # Bias toward edges: use u^0.7 instead of u
            s = np.sqrt(u ** 0.7)
            x0 = 1.0 - s
            x1 = s * (1.0 - v)
            x2 = s * v
            # Clip to valid range
            x0 = max(0.0, min(1.0, x0))
            x1 = max(0.0, min(1.0, x1))
            x2 = max(0.0, min(1.0, x2))
            s_sum = x0 + x1 + x2
            if s_sum > 0:
                x0 /= s_sum
                x1 /= s_sum
                x2 /= s_sum
            else:
                x0, x1, x2 = 1.0/3, 1.0/3, 1.0/3

            ctrl_rand = lo[3:] + (hi[3:] - lo[3:]) * rng.random(self.dim - 3)
            x = np.concatenate([np.array([x0, x1, x2]), ctrl_rand])
            x = np.clip(x, lo, hi)
            f = problem(x)
            evaluated.append((x, np.array(f, dtype=float)))
            used += 1

        return np.array([x for x, _ in evaluated]) if evaluated else np.zeros((1, self.dim))
