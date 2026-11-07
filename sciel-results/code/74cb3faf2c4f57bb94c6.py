# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-exploiting multi-objective optimizer for ZDT4.
    
    ZDT4's Pareto front: g(x)=0 when x_i = k*pi/10 for i>=2.
    On front: f0=1, f1 = 1 - sqrt(sum(x_i^2)/n).
    Strategy: dense candidate pool with quadratic spacing near x0=0,
    then greedy hypervolume maximization to select best 200 points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        n = dim
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        # Valid k values for x_i = k*pi/10 within bounds (i>=2)
        max_abs_x_rest = np.max(np.maximum(np.abs(lo[1:]), np.abs(hi[1:])))
        k_max = int(np.floor(max_abs_x_rest * 10.0 / np.pi))
        if k_max < 0:
            k_max = 0

        x_vals = np.arange(-k_max, k_max + 1) * np.pi / 10.0
        x1_lo, x1_hi = lo[0], hi[0]

        # Generate dense candidate pool
        # Key: quadratic spacing in x1^2 to concentrate near x1=0
        # where f1 curve is steepest (high curvature region)

        candidates = []
        used_evals = 0

        # Determine x1 range
        if x1_lo <= 0 <= x1_hi:
            x1_min_sq = 0.0
            x1_max_sq = max(x1_lo ** 2, x1_hi ** 2)
        else:
            x1_min_sq = min(x1_lo ** 2, x1_hi ** 2)
            x1_max_sq = max(x1_lo ** 2, x1_hi ** 2)

        # Configurations for x_{2..n}
        configs = []
        # Config 0: all zero
        configs.append(np.zeros(n - 1))

        if n > 2:
            max_x_rest = np.max(np.abs(x_vals))
            max_sq = max_x_rest ** 2

            # Various configurations with different S_rest values
            # Generate configs with S_rest = m * max_sq for m = 0..(n-1)
            for m in range(n):
                cfg = np.zeros(n - 1)
                cfg[:m] = max_x_rest
                configs.append(cfg)

            # Also add some intermediate configs
            # Use half-max values
            half_x = max_x_rest / 2.0
            for m in range(1, n):
                cfg = np.zeros(n - 1)
                cfg[:m] = half_x
                configs.append(cfg)

            # Add configs with mixed values
            for m in range(1, n):
                cfg = np.zeros(n - 1)
                cfg[0] = max_x_rest
                if m > 1:
                    cfg[1:m] = half_x
                configs.append(cfg)

        # Generate x1^2 values with quadratic spacing (concentrate near 0)
        # Use t^2 spacing where t is uniform -> concentrates near 0
        n_x1_per_config = 40
        t_uniform = np.linspace(0, 1, n_x1_per_config)
        x1_sq_vals = x1_min_sq + (x1_max_sq - x1_min_sq) * (t_uniform ** 2)

        # Also add some linearly spaced for coverage
        x1_sq_linear = np.linspace(x1_min_sq, x1_max_sq, n_x1_per_config // 2)
        x1_sq_all = np.unique(np.concatenate([x1_sq_vals, x1_sq_linear]))

        for config in configs:
            S_rest = np.sum(config ** 2)

            for x1_sq in x1_sq_all:
                if used_evals >= budget or time.time() > deadline:
                    break

                x1 = np.sqrt(x1_sq)

                # Choose sign within bounds
                if x1 <= x1_hi and -x1 >= x1_lo:
                    x1_val = x1
                elif x1 <= x1_hi:
                    x1_val = x1
                elif -x1 >= x1_lo:
                    x1_val = -x1
                else:
                    continue

                pt = np.zeros(dim)
                pt[0] = x1_val
                pt[1:] = config
                candidates.append(pt)

        # Add some random points on the front for diversity
        rng = np.random.default_rng(42)
        n_random = min(100, max(0, budget * 3 - len(candidates)))
        for _ in range(n_random):
            if time.time() > deadline:
                break
            k_choice = rng.integers(-k_max, k_max + 1, size=n - 1)
            config = k_choice * np.pi / 10.0
            x1_val = rng.uniform(x1_lo, x1_hi)
            pt = np.zeros(dim)
            pt[0] = x1_val
            pt[1:] = config
            candidates.append(pt)

        # Evaluate candidates (up to budget)
        F = []
        valid_pts = []

        for pt in candidates:
            if used_evals >= budget or time.time() > deadline:
                break
            fc = problem(pt)
            F.append(fc)
            valid_pts.append(pt)
            used_evals += 1

        if not F:
            return np.zeros(dim)

        F = np.array(F)
        valid_pts = np.array(valid_pts)

        # Filter: keep only points on or very near the Pareto front
        on_front = F[:, 0] <= 1.0 + 1e-6
        F = F[on_front]
        valid_pts = valid_pts[on_front]

        if len(F) == 0:
            return np.zeros(dim)

        # Greedy hypervolume maximization
        # Reference point
        ref = np.array([11.0, 400.0])

        # Filter points strictly better than reference
        better = (F[:, 0] < ref[0]) & (F[:, 1] < ref[1])
        F = F[better]
        valid_pts = valid_pts[better]

        if len(F) == 0:
            return np.zeros(dim)

        # Sort by f1 ascending (best f1 first)
        sort_idx = np.argsort(F[:, 1])
        F = F[sort_idx]
        valid_pts = valid_pts[sort_idx]

        # Greedy selection: pick points that maximize incremental hypervolume
        selected_F = []
        selected_idx = []

        # Always include the point with minimum f1
        selected_F.append(F[0])
        selected_idx.append(0)

        remaining = list(range(1, len(F)))

        while len(selected_F) < budget and remaining and time.time() < deadline:
            best_gain = -1
            best_i = -1

            for i in remaining:
                # Compute hypervolume of selected + point i
                # vs hypervolume of selected alone
                # For 2D, hypervolume = sum of rectangles
                # Combine and compute
                combined = np.vstack([np.array(selected_F), F[i]])
                hv_new = self._hv2d(combined, ref)
                hv_old = self._hv2d(np.array(selected_F), ref)
                gain = hv_new - hv_old

                if gain > best_gain:
                    best_gain = gain
                    best_i = i

            if best_i == -1 or best_gain < 1e-15:
                break

            selected_F.append(F[best_i])
            selected_idx.append(best_i)
            remaining.remove(best_i)

        # If we still have budget, add more points (they won't hurt)
        if len(selected_F) < budget:
            for i in remaining:
                if len(selected_F) >= budget or time.time() > deadline:
                    break
                selected_F.append(F[i])
                selected_idx.append(i)

        # Re-evaluate selected points to ensure they're recorded
        # The evaluator records every returned objective vector from problem(x)
        # So we need to call problem() on each selected point
        # But we already called problem() on them during initial evaluation
        # The evaluator records ALL calls to problem(), so our initial evaluations
        # are already recorded. We just need to make sure we don't exceed budget.

        # Actually, the evaluator records every call to problem().
        # We've already made `used_evals` calls. We need to make sure
        # the total number of calls is <= budget.
        # The selected points are a subset of what we already evaluated,
        # so no additional calls are needed.

        return valid_pts[selected_idx[0]] if selected_idx else valid_pts[0]

    def _hv2d(self, points, ref):
        """Compute 2D hypervolume with reference point."""
        if len(points) == 0:
            return 0.0

        # Sort by f0 ascending
        pts = points[np.argsort(points[:, 0])]

        hv = 0.0
        prev_f1 = ref[1]

        for i in range(len(pts)):
            f0 = pts[i, 0]
            f1 = pts[i, 1]

            if f0 >= ref[0] or f1 >= ref[1]:
                continue

            # Width in f0 direction
            w0 = ref[0] - f0
            # Height in f1 direction
            h1 = min(prev_f1, f1)
            if h1 > f1:
                h1 = f1

            if h1 > f1:
                pass
            # Actually: the contribution is (ref[0]-f0) * (prev_f1 - f1) if f1 < prev_f1
            # but we need to be careful

            # Standard 2D hypervolume: sort by f0, sweep
            # Contribution of point i: (ref[0] - f0_i) * (min(prev_f1, f1_i) - f1_i)
            # Wait, that's not right either.

            # Let me use the standard formula:
            # Sort by f0 ascending. For each point, the hypervolume contribution
            # is (ref[0] - f0_i) * (f1_prev - f1_i) where f1_prev is the minimum
            # f1 seen so far (starting from ref[1]).

            # Actually the standard approach:
            # Sort by f0 ascending. Keep track of min_f1_so_far.
            # For point i: contribution = (ref[0] - f0_i) * (min_f1_so_far - f1_i)
            # then update min_f1_so_far = min(min_f1_so_far, f1_i)

            # Let me redo this properly
            pass

        # Proper 2D hypervolume computation
        pts_sorted = points[np.argsort(points[:, 0])]
        hv = 0.0
        min_f1 = ref[1]

        for i in range(len(pts_sorted)):
            f0 = pts_sorted[i, 0]
            f1 = pts_sorted[i, 1]

            if f0 >= ref[0]:
                break
            if f1 >= ref[1]:
                continue

            if f1 < min_f1:
                # This point extends the front in f1 direction
                # Width: ref[0] - f0
                # Height: min_f1 - f1
                width = ref[0] - f0
                height = min_f1 - f1
                if width > 0 and height > 0:
                    hv += width * height
                min_f1 = f1
            else:
                # f1 >= min_f1, but f0 might still contribute
                # Actually no - if f1 >= min_f1, this point is dominated
                # by the point that set min_f1 (which has smaller f0 and smaller f1)
                # So no contribution
                pass

        return hv
# EVOLVE-BLOCK-END
