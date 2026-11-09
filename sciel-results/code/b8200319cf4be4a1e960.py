"""Multi-objective optimizer for RE21 using stratified sampling + adaptive local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


class Optimizer:
    """Multi-objective optimizer for RE21 (4 vars, 2 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

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

        def time_ok():
            return time.time() < deadline

        # Phase 1: Stratified grid sampling (3^4 = 81 points)
        n_per_dim = 3
        levels = [np.linspace(lo[d], hi[d], n_per_dim) for d in range(self.dim)]
        for combo in itertools.product(*levels):
            if used >= 100 or not time_ok():
                break
            x = np.array(combo, dtype=np.float64)
            eval_point(x)

        # Fill remaining phase-1 budget with random points
        while used < 100 and time_ok():
            x = lo + span * np.random.random(self.dim)
            eval_point(x)

        # Phase 2: Adaptive local refinement around non-dominated points
        if not all_F:
            return np.array([])

        all_F_arr = np.array(all_F)
        all_X_arr = np.array(all_X)

        # Filter to valid points (better than ref in all objectives)
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

        # Sort non-dominated points by first objective for diversity
        order = np.argsort(nd_F[:, 0])
        nd_F = nd_F[order]
        nd_X = nd_X[order]

        # Phase 2: Local search around non-dominated points
        remaining = self.budget - used
        if remaining > 0 and time_ok():
            # Adaptive step size based on local density
            for i in range(len(nd_X)):
                if used >= self.budget or not time_ok():
                    break

                x_center = nd_X[i]
                # Estimate local scale from neighbors
                if len(nd_X) > 1:
                    idx = (i + 1) % len(nd_X)
                    local_scale = np.linalg.norm(nd_X[idx] - x_center)
                    if local_scale < 1e-10:
                        local_scale = 0.05 * np.linalg.norm(span)
                    step = max(0.02 * np.linalg.norm(span), 0.5 * local_scale)
                else:
                    step = 0.05 * np.linalg.norm(span)

                # Multi-directional local search
                n_attempts = min(8, self.budget - used)
                for _ in range(n_attempts):
                    if used >= self.budget or not time_ok():
                        break
                    # Random direction with adaptive step
                    direction = np.random.normal(0, 1, self.dim)
                    direction /= (np.linalg.norm(direction) + 1e-15)
                    x_new = x_center + step * direction
                    x_new = np.clip(x_new, lo, hi)
                    eval_point(x_new)

                # Axis-aligned fine search
                for d in range(self.dim):
                    if used >= self.budget or not time_ok():
                        break
                    for direction in [-1, 1]:
                        if used >= self.budget or not time_ok():
                            break
                        x_new = x_center.copy()
                        x_new[d] = x_center[d] + direction * 0.03 * span[d]
                        x_new = np.clip(x_new, lo, hi)
                        eval_point(x_new)

        # Phase 3: Final polish - refine top hypervolume contributors
        if used < self.budget and time_ok():
            all_F_arr = np.array(all_F)
            all_X_arr = np.array(all_X)
            valid_mask = np.all(all_F_arr < ref_point, axis=1)
            if np.any(valid_mask):
                valid_F = all_F_arr[valid_mask]
                valid_X = all_X_arr[valid_mask]
            else:
                valid_F = all_F_arr
                valid_X = all_X_arr

            if len(valid_F) > 0:
                nd_mask = self._non_dominated_mask(valid_F)
                nd_F = valid_F[nd_mask]
                nd_X = valid_X[nd_mask]

                # Compute contribution proxy: distance to reference point
                # Points closer to origin (in objective space) contribute more
                distances = np.sum((ref_point - nd_F) ** 2, axis=1)
                top_k = min(5, len(nd_X))
                top_indices = np.argsort(distances)[:top_k]

                for idx in top_indices:
                    if used >= self.budget or not time_ok():
                        break
                    x_center = nd_X[idx]
                    # Very fine local search
                    for _ in range(4):
                        if used >= self.budget or not time_ok():
                            break
                        x_new = x_center + np.random.normal(0, 0.01, self.dim) * span
                        x_new = np.clip(x_new, lo, hi)
                        eval_point(x_new)

        # Return all evaluated points (evaluator records all objective vectors)
        return np.array(all_X) if all_X else np.array([])

    def _non_dominated_mask(self, F):
        """Return a boolean mask for non-dominated points in F (minimization)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)

        nd_mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd_mask[i]:
                continue
            for j in range(n):
                if i == j or not nd_mask[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd_mask[i] = False
                    break
        return nd_mask
# EVOLVE-BLOCK-END
