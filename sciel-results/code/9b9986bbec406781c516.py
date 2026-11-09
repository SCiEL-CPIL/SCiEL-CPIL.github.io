"""Multi-objective optimizer for RE21 using hypervolume-based selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using hypervolume indicator for selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref_point = np.array([3000.0, 0.06])
        rng = np.random.default_rng(42)

        # Archive size limit
        max_archive = 100
        initial_pop = 20

        # Initialize
        X_archive = lo + (hi - lo) * rng.random((initial_pop, self.dim))
        F_archive = np.array([problem(x) for x in X_archive], dtype=np.float64)
        used = initial_pop

        # Filter to non-dominated
        X_archive, F_archive = self._prune_archive(X_archive, F_archive, ref_point, max_archive)

        # Track progress for adaptive sigma
        total_budget = self.budget
        initial_sigma_scale = 0.3
        final_sigma_scale = 0.02

        while used < self.budget:
            if time.time() > deadline - 5:
                break

            # Adaptive sigma: start large, end small
            progress = used / total_budget
            sigma_scale = initial_sigma_scale + (final_sigma_scale - initial_sigma_scale) * progress
            sigma = sigma_scale * (hi - lo)

            # Generate offspring
            n_offspring = 20
            offspring_X = np.empty((n_offspring, self.dim))
            offspring_F = np.empty((n_offspring, self.n_obj))
            valid_offspring = 0

            for i in range(n_offspring):
                if time.time() > deadline - 5:
                    break
                if used >= self.budget:
                    break

                # Mix of exploration and exploitation
                if rng.random() < 0.25:
                    # Exploration: random point
                    child = lo + (hi - lo) * rng.random(self.dim)
                else:
                    # Exploitation: mutate a point from archive
                    # Select parent with probability proportional to hypervolume contribution
                    parent_idx = self._select_parent_hype(F_archive, ref_point, rng)
                    parent = X_archive[parent_idx]
                    # Gaussian mutation with adaptive sigma
                    child = parent + rng.normal(0, 1, self.dim) * sigma
                    child = np.clip(child, lo, hi)

                offspring_X[valid_offspring] = child
                offspring_F[valid_offspring] = problem(child)
                used += 1
                valid_offspring += 1

            if valid_offspring == 0:
                break

            # Trim arrays to valid offspring
            offspring_X = offspring_X[:valid_offspring]
            offspring_F = offspring_F[:valid_offspring]

            # Combine
            combined_X = np.vstack([X_archive, offspring_X])
            combined_F = np.vstack([F_archive, offspring_F])

            # Prune to non-dominated and limit size
            X_archive, F_archive = self._prune_archive(combined_X, combined_F, ref_point, max_archive)

        # Local search polish: use remaining budget to refine best points
        remaining = self.budget - used
        if remaining > 10 and time.time() < deadline - 10:
            X_archive, F_archive = self._local_search_polish(
                problem, X_archive, F_archive, lo, hi, ref_point, 
                remaining, deadline, rng, max_archive
            )

        return X_archive

    def _select_parent_hype(self, F, ref_point, rng):
        """Select parent with probability proportional to hypervolume contribution."""
        n = len(F)
        if n == 1:
            return 0

        # Compute hypervolume contributions in 2D
        sort_idx = np.argsort(F[:, 0])
        F_sorted = F[sort_idx]

        contributions = np.zeros(n)
        for i in range(n - 1):
            w = F_sorted[i+1, 0] - F_sorted[i, 0]
            h = ref_point[1] - F_sorted[i, 1]
            contributions[sort_idx[i]] = max(0, w) * max(0, h)
        w = ref_point[0] - F_sorted[-1, 0]
        h = ref_point[1] - F_sorted[-1, 1]
        contributions[sort_idx[-1]] = max(0, w) * max(0, h)

        # Normalize and sample
        total = contributions.sum()
        if total <= 0:
            return rng.integers(n)
        probs = contributions / total
        return rng.choice(n, p=probs)

    def _local_search_polish(self, problem, X, F, lo, hi, ref_point, 
                             budget, deadline, rng, max_archive):
        """Local search polish: refine each archive point with small steps."""
        n_points = len(X)
        if n_points == 0 or budget < n_points * 2:
            return X, F

        # Reserve budget for polishing
        polish_budget = min(budget, n_points * 5)
        steps_per_point = max(1, polish_budget // n_points)

        for i in range(n_points):
            if time.time() > deadline - 5:
                break
            if polish_budget <= 0:
                break

            x = X[i].copy()
            f = F[i].copy()
            best_x = x.copy()
            best_f = f.copy()

            for step in range(steps_per_point):
                if time.time() > deadline - 5:
                    break
                if polish_budget <= 0:
                    break

                # Try small perturbations in each dimension
                for dim in range(self.dim):
                    if polish_budget <= 0:
                        break
                    if time.time() > deadline - 5:
                        break

                    # Small step size
                    step_size = 0.01 * (hi[dim] - lo[dim])
                    for direction in [-1, 1]:
                        if polish_budget <= 0:
                            break
                        if time.time() > deadline - 5:
                            break

                        trial_x = x.copy()
                        trial_x[dim] += direction * step_size
                        trial_x = np.clip(trial_x, lo, hi)

                        trial_f = problem(trial_x)
                        polish_budget -= 1

                        # Check if trial dominates current best
                        if np.all(trial_f <= best_f) and np.any(trial_f < best_f):
                            best_x = trial_x.copy()
                            best_f = trial_f.copy()
                            x = trial_x.copy()
                            f = trial_f.copy()
                        elif np.all(trial_f <= f) and np.any(trial_f < f):
                            # Better than current but not best, move here
                            x = trial_x.copy()
                            f = trial_f.copy()

            # Update archive with improved point
            X[i] = best_x
            F[i] = best_f

        # Final prune
        X, F = self._prune_archive(X, F, ref_point, max_archive)
        return X, F

    def _prune_archive(self, X, F, ref_point, max_size):
        """Prune archive to non-dominated points and limit size by hypervolume contribution."""
        n = len(X)
        if n == 0:
            return X, F

        # 1. Remove dominated points
        non_dom_mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not non_dom_mask[i]:
                continue
            for j in range(n):
                if i == j or not non_dom_mask[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    non_dom_mask[i] = False
                    break

        X_nd = X[non_dom_mask]
        F_nd = F[non_dom_mask]

        if len(X_nd) <= max_size:
            return X_nd, F_nd

        # 2. If too many, prune based on hypervolume contribution
        # For 2D, we can compute contribution efficiently
        # Sort by first objective
        sort_idx = np.argsort(F_nd[:, 0])
        X_sorted = X_nd[sort_idx]
        F_sorted = F_nd[sort_idx]

        contributions = np.zeros(len(F_sorted))
        # Hypervolume contribution in 2D:
        # A point (f1, f2) contributes the area of the rectangle
        # [f1, next_f1] x [f2, ref2] if it's on the Pareto front
        # Actually, simpler: sort by f1 ascending.
        # The contribution of point i is (f1[i+1] - f1[i]) * (ref2 - f2[i])
        # But we need to handle the last point and ensure we only count non-dominated parts.
        # Since they are all non-dominated, sorting by f1 ascending implies f2 descending.

        # Check if strictly non-dominated and sorted
        # f1 increasing, f2 decreasing
        for i in range(len(F_sorted) - 1):
            # Width in f1
            w = F_sorted[i+1, 0] - F_sorted[i, 0]
            if w <= 0: w = 0
            # Height in f2
            h = ref_point[1] - F_sorted[i, 1]
            if h <= 0: h = 0
            contributions[i] = w * h

        # Last point
        w = ref_point[0] - F_sorted[-1, 0]
        if w <= 0: w = 0
        h = ref_point[1] - F_sorted[-1, 1]
        if h <= 0: h = 0
        contributions[-1] = w * h

        # Remove points with smallest contributions
        if len(contributions) > max_size:
            # Keep top max_size contributions
            # But we must be careful: removing a point might change the contribution of neighbors.
            # For a quick prune, just removing the smallest is a heuristic.
            # A better way is to iteratively remove the point with the smallest contribution.
            # Given max_size is 100 and n is small, iterative removal is fine.
            keep_indices = np.arange(len(F_sorted))
            current_contributions = contributions.copy()
            current_F = F_sorted.copy()
            current_X = X_sorted.copy()
            
            while len(keep_indices) > max_size:
                # Find index of min contribution
                min_idx = np.argmin(current_contributions)
                # Remove it
                keep_indices = np.delete(keep_indices, min_idx)
                
                # Recalculate contributions for the remaining set
                # This is O(N^2) total which is fine for N=100-200
                F_rem = current_F[keep_indices]
                X_rem = current_X[keep_indices]
                
                # Re-sort and recalc
                sort_idx_rem = np.argsort(F_rem[:, 0])
                F_rem_sorted = F_rem[sort_idx_rem]
                
                new_contribs = np.zeros(len(F_rem_sorted))
                for i in range(len(F_rem_sorted) - 1):
                    w = F_rem_sorted[i+1, 0] - F_rem_sorted[i, 0]
                    h = ref_point[1] - F_rem_sorted[i, 1]
                    new_contribs[i] = max(0, w) * max(0, h)
                w = ref_point[0] - F_rem_sorted[-1, 0]
                h = ref_point[1] - F_rem_sorted[-1, 1]
                new_contribs[-1] = max(0, w) * max(0, h)
                
                current_contributions = new_contribs
                current_F = F_rem_sorted
                current_X = X_rem[sort_idx_rem]
                keep_indices = np.arange(len(F_rem_sorted))
            
            return current_X, current_F

        return X_nd, F_nd
# EVOLVE-BLOCK-END
