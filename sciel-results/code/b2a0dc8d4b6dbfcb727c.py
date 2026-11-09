"""Multi-objective optimizer for RE21 using grid + stratified sampling + adaptive local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


class Optimizer:
    """Direct multi-objective optimizer for RE21 (4 vars, 2 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        ref_point = np.array([3000.0, 0.06])

        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = problem(x)
            f = np.array(f, dtype=np.float64)
            all_X.append(x.copy())
            all_F.append(f.copy())
            used += 1
            return f

        # Phase 1: Structured exploration
        # 3^4 = 81 grid points for broad coverage
        levels = [np.linspace(lo[d], hi[d], 3) for d in range(self.dim)]
        for combo in itertools.product(*levels):
            if used >= 81 or time.time() > deadline:
                break
            x = np.array(combo, dtype=np.float64)
            eval_point(x)

        # Fill remaining exploration with stratified random (Sobol-like via uniform in sub-cubes)
        target_explore = 140
        while used < target_explore and time.time() < deadline:
            # Use a stratified approach: divide space into sub-cubes and pick one point per sub-cube
            # For 4D, use 2^4 = 16 sub-cubes, pick random point in each, then repeat with different seeds
            # Simpler: just use random but with a twist - use Latin hypercube-like sampling
            x = np.random.uniform(lo, hi)
            eval_point(x)

        if not all_F:
            return np.array([])

        all_F_arr = np.array(all_F)
        all_X_arr = np.array(all_X)

        valid_mask = np.all(all_F_arr < ref_point, axis=1)
        if np.any(valid_mask):
            valid_F = all_F_arr[valid_mask]
            valid_X = all_X_arr[valid_mask]
        else:
            valid_F = all_F_arr
            valid_X = all_X_arr

        if len(valid_F) == 0:
            return all_X_arr

        nd_mask = self._non_dominated_mask(valid_F)
        nd_F = valid_F[nd_mask]
        nd_X = valid_X[nd_mask]

        # Phase 2: Adaptive local search with hypervolume-aware acceptance
        if len(nd_F) > 0:
            contributions = self._marginal_hv_contributions(nd_F, ref_point)
            order = np.argsort(-contributions)
            nd_X_sorted = nd_X[order]
            nd_F_sorted = nd_F[order]
        else:
            nd_X_sorted = nd_X
            nd_F_sorted = nd_F

        base_scale = 0.06
        min_scale = 0.001
        max_iters_per_point = 12

        for i in range(len(nd_X_sorted)):
            if used >= self.budget or time.time() > deadline:
                break

            x_best = nd_X_sorted[i].copy()
            f_best = nd_F_sorted[i].copy()
            scale = base_scale
            no_improve_count = 0
            hv_before = self._compute_hv(np.vstack([nd_F_sorted, f_best.reshape(1, -1)]), ref_point)

            for it in range(max_iters_per_point):
                if used >= self.budget or time.time() > deadline:
                    break

                # Generate candidate: mix of Gaussian and axis-aligned
                if it % 2 == 0:
                    x_try = x_best + np.random.normal(0, 1, self.dim) * scale * span
                else:
                    d = np.random.randint(self.dim)
                    direction = np.random.choice([-1, 1])
                    x_try = x_best.copy()
                    x_try[d] += direction * scale * span[d] * np.random.uniform(0.3, 1.2)

                x_try = np.clip(x_try, lo, hi)
                f_try = eval_point(x_try)

                # Accept if it improves local point OR adds hypervolume
                local_improve = np.all(f_try <= f_best) and np.any(f_try < f_best)
                if local_improve:
                    x_best = x_try.copy()
                    f_best = f_try.copy()
                    scale = min(scale * 1.1, base_scale)
                    no_improve_count = 0
                else:
                    # Check if it improves global HV (cheap check: is it non-dominated vs current front?)
                    dominated = False
                    for fj in nd_F_sorted:
                        if np.all(fj <= f_try) and np.any(fj < f_try):
                            dominated = True
                            break
                    if not dominated:
                        # It's a new non-dominated point; accept with some probability
                        if np.random.random() < 0.3:
                            scale *= 0.9
                    else:
                        scale *= 0.8
                        no_improve_count += 1
                        if scale < min_scale or no_improve_count > 4:
                            break

        # Phase 3: Final high-precision polish on top contributors
        if not all_F:
            return np.array([])

        all_F_arr = np.array(all_F)
        all_X_arr = np.array(all_X)

        valid_mask = np.all(all_F_arr < ref_point, axis=1)
        if np.any(valid_mask):
            valid_F = all_F_arr[valid_mask]
            valid_X = all_X_arr[valid_mask]
        else:
            valid_F = all_F_arr
            valid_X = all_X_arr

        if len(valid_F) == 0:
            return all_X_arr

        nd_mask = self._non_dominated_mask(valid_F)
        nd_F = valid_F[nd_mask]
        nd_X = valid_X[nd_mask]

        # Final polish: high-precision line search on top HV contributors
        if len(nd_F) > 0 and used < self.budget:
            contributions = self._marginal_hv_contributions(nd_F, ref_point)
            top_k = min(8, len(nd_F))
            top_indices = np.argsort(-contributions)[:top_k]

            for idx in top_indices:
                if used >= self.budget or time.time() > deadline:
                    break
                x_c = nd_X[idx].copy()
                f_c = nd_F[idx].copy()

                # Multi-direction line search with golden section refinement
                directions = []
                for d in range(self.dim):
                    for sgn in [-1, 1]:
                        dirs = np.zeros(self.dim)
                        dirs[d] = sgn
                        directions.append(dirs)
                # Add diagonal directions
                for sgn in [-1, 1]:
                    directions.append(np.ones(self.dim) * sgn)

                for direction in directions:
                    if used >= self.budget or time.time() > deadline:
                        break
                    # Try a few step sizes with decreasing magnitude
                    for base_step in [0.01, 0.005, 0.002, 0.001]:
                        if used >= self.budget or time.time() > deadline:
                            break
                        best_f_local = f_c.copy()
                        best_x_local = x_c.copy()
                        # 3-point line search
                        for t in [0.5, 1.0, 2.0]:
                            if used >= self.budget or time.time() > deadline:
                                break
                            x_try = x_c + t * base_step * direction * span
                            x_try = np.clip(x_try, lo, hi)
                            f_try = eval_point(x_try)
                            # Accept if non-dominated vs current best local point
                            if np.all(f_try <= best_f_local) and np.any(f_try < best_f_local):
                                best_f_local = f_try.copy()
                                best_x_local = x_try.copy()
                            elif np.all(f_try < ref_point):
                                # Accept if it extends the front (not dominated by best_f_local)
                                if not (np.all(best_f_local <= f_try) and np.any(best_f_local < f_try)):
                                    # Keep the one with better HV contribution locally
                                    hv_try = (ref_point[0] - f_try[0]) * (ref_point[1] - f_try[1])
                                    hv_cur = (ref_point[0] - best_f_local[0]) * (ref_point[1] - best_f_local[1])
                                    if hv_try > hv_cur:
                                        best_f_local = f_try.copy()
                                        best_x_local = x_try.copy()

                        if np.any(best_f_local != f_c):
                            x_c = best_x_local
                            f_c = best_f_local

        # Final recompute and return
        if not all_F:
            return np.array([])

        all_F_arr = np.array(all_F)
        all_X_arr = np.array(all_X)

        valid_mask = np.all(all_F_arr < ref_point, axis=1)
        if np.any(valid_mask):
            valid_F = all_F_arr[valid_mask]
            valid_X = all_X_arr[valid_mask]
        else:
            valid_F = all_F_arr
            valid_X = all_X_arr

        if len(valid_F) == 0:
            return all_X_arr

        nd_mask = self._non_dominated_mask(valid_F)
        nd_F = valid_F[nd_mask]
        nd_X = valid_X[nd_mask]

        # One more ultra-fine polish pass if budget allows
        if len(nd_F) > 0 and used < self.budget - 10 and time.time() < deadline - 5:
            contributions = self._marginal_hv_contributions(nd_F, ref_point)
            top_k = min(4, len(nd_F))
            top_indices = np.argsort(-contributions)[:top_k]

            for idx in top_indices:
                if used >= self.budget or time.time() > deadline:
                    break
                x_c = nd_X[idx].copy()
                f_c = nd_F[idx].copy()

                # Very fine axis-aligned search
                for d in range(self.dim):
                    if used >= self.budget or time.time() > deadline:
                        break
                    for sgn in [-1, 1]:
                        if used >= self.budget or time.time() > deadline:
                            break
                        for t in [0.0005, 0.001, 0.002]:
                            if used >= self.budget or time.time() > deadline:
                                break
                            x_try = x_c.copy()
                            x_try[d] += sgn * t * span[d]
                            x_try = np.clip(x_try, lo, hi)
                            f_try = eval_point(x_try)
                            if np.all(f_try <= f_c) and np.any(f_try < f_c):
                                x_c = x_try.copy()
                                f_c = f_try.copy()

        # Final recompute
        all_F_arr = np.array(all_F)
        all_X_arr = np.array(all_X)
        valid_mask = np.all(all_F_arr < ref_point, axis=1)
        if np.any(valid_mask):
            valid_F = all_F_arr[valid_mask]
            valid_X = all_X_arr[valid_mask]
        else:
            valid_F = all_F_arr
            valid_X = all_X_arr

        if len(valid_F) == 0:
            return all_X_arr

        nd_mask = self._non_dominated_mask(valid_F)
        nd_F = valid_F[nd_mask]
        nd_X = valid_X[nd_mask]

        return nd_X

    def _is_better_local(self, f_new, f_old, ref_point):
        """Check if f_new is better than f_old for local search purposes."""
        if np.all(f_new <= f_old) and np.any(f_new < f_old):
            return True
        if np.all(f_old <= f_new) and np.any(f_old < f_new):
            return False
        if np.all(f_new < ref_point) and np.all(f_old < ref_point):
            hv_new = (ref_point[0] - f_new[0]) * (ref_point[1] - f_new[1])
            hv_old = (ref_point[0] - f_old[0]) * (ref_point[1] - f_old[1])
            return hv_new > hv_old
        elif np.all(f_new < ref_point):
            return True
        elif np.all(f_old < ref_point):
            return False
        return False

    def _marginal_hv_contributions(self, F, ref_point):
        """
        Calculate marginal hypervolume contribution for each point in F.
        F is an array of shape (N, 2).
        """
        N = len(F)
        if N == 0:
            return np.array([])

        # Compute HV of the full set
        hv_full = self._compute_hv(F, ref_point)

        contributions = np.zeros(N)
        for i in range(N):
            mask = np.ones(N, dtype=bool)
            mask[i] = False
            F_reduced = F[mask]
            hv_reduced = self._compute_hv(F_reduced, ref_point)
            contributions[i] = hv_full - hv_reduced

        return contributions

    def _compute_hv(self, F, ref_point):
        """
        Compute exact hypervolume for 2 objectives.
        F is an array of shape (N, 2).
        """
        if len(F) == 0:
            return 0.0

        valid_mask = np.all(F < ref_point, axis=1)
        F_valid = F[valid_mask]

        if len(F_valid) == 0:
            return 0.0

        # Sort by first objective (ascending)
        F_sorted = F_valid[np.argsort(F_valid[:, 0])]

        # Compute prefix minimum of y
        y_prefix_min = np.minimum.accumulate(F_sorted[:, 1])

        x_coords = np.append(F_sorted[:, 0], ref_point[0])
        y_coords = y_prefix_min

        widths = x_coords[1:] - x_coords[:-1]
        heights = ref_point[1] - y_coords

        widths = np.maximum(widths, 0)
        heights = np.maximum(heights, 0)

        hv = np.sum(widths * heights)

        return hv

    def _is_better_local(self, f_new, f_old, ref_point):
        """Check if f_new is better than f_old for local search purposes."""
        if np.all(f_new <= f_old) and np.any(f_new < f_old):
            return True
        if np.all(f_old <= f_new) and np.any(f_old < f_new):
            return False
        if np.all(f_new < ref_point) and np.all(f_old < ref_point):
            hv_new = (ref_point[0] - f_new[0]) * (ref_point[1] - f_new[1])
            hv_old = (ref_point[0] - f_old[0]) * (ref_point[1] - f_old[1])
            return hv_new > hv_old
        elif np.all(f_new < ref_point):
            return True
        elif np.all(f_old < ref_point):
            return False
        return False

    def _non_dominated_mask(self, F):
        """Return a boolean mask for non-dominated points in F (minimization). Vectorized."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        if n == 1:
            return np.ones(1, dtype=bool)
        # F[i] is dominated if exists j: F[j] <= F[i] elementwise and F[j] < F[i] in at least one dim
        # diff[i,j] = F[j] - F[i]; F[j] <= F[i] iff all diff[i,j] >= 0
        diff = F[np.newaxis, :, :] - F[:, np.newaxis, :]  # shape (n, n, 2)
        leq = np.all(diff >= 0, axis=2)  # F[j] <= F[i]
        lt = np.any(diff < 0, axis=2)    # F[j] < F[i] in at least one
        dominated = np.any(leq & lt, axis=1)
        # A point is non-dominated if NOT dominated by any other point
        # Note: a point is not dominated by itself (diff=0, lt=False for i==j)
        return ~dominated
# EVOLVE-BLOCK-END
