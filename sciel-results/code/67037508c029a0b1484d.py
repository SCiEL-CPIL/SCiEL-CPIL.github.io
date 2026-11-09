"""Multi-objective optimizer for RE21 using hypervolume-based selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using HypE-style marginal hypervolume selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref_point = np.array([3000.0, 0.06])
        rng = np.random.default_rng(42)

        # Archive size limit
        max_archive = 100
        initial_pop = 20

        # Stratified initialization for better spread
        X_archive = self._stratified_sample(rng, lo, hi, initial_pop, self.dim)
        F_archive = np.array([problem(x) for x in X_archive], dtype=np.float64)
        used = initial_pop

        # Filter to non-dominated
        X_archive, F_archive = self._prune_archive(X_archive, F_archive, ref_point, max_archive)

        # Adaptive sigma: start wide, decay over time
        base_sigma = 0.12
        sigma_decay = 0.97

        # Track best HV so far for early stopping
        best_hv = self._compute_hypervolume(F_archive, ref_point)

        while used < self.budget:
            if time.time() > deadline - 5:
                break

            # Generate offspring
            n_offspring = 20
            offspring_X = np.empty((n_offspring, self.dim))
            offspring_F = np.empty((n_offspring, self.n_obj))

            for i in range(n_offspring):
                if time.time() > deadline - 5:
                    break
                if used >= self.budget:
                    break

                # Mix of exploration and exploitation
                if rng.random() < 0.20:
                    # Exploration: random point
                    child = lo + (hi - lo) * rng.random(self.dim)
                else:
                    # Exploitation: select parent by marginal hypervolume contribution
                    parent_idx = self._select_parent_hype(F_archive, ref_point, rng)
                    parent = X_archive[parent_idx]
                    # Adaptive sigma with per-dimension scaling
                    progress = used / max(1, self.budget)
                    sigma = base_sigma * (1.0 - 0.5 * progress)
                    # Per-dimension sigma scaled by range
                    sigma_vec = sigma * (hi - lo)
                    # Add some L1 noise for better exploration in high-dim
                    child = parent + rng.normal(0, 1, self.dim) * sigma_vec
                    # Occasionally use larger jumps for escape
                    if rng.random() < 0.1:
                        child = parent + rng.normal(0, 1, self.dim) * sigma_vec * 2.0
                    child = np.clip(child, lo, hi)

                offspring_X[i] = child
                offspring_F[i] = problem(child)
                used += 1

            # Combine
            combined_X = np.vstack([X_archive, offspring_X])
            combined_F = np.vstack([F_archive, offspring_F])

            # Prune to non-dominated and limit size
            X_archive, F_archive = self._prune_archive(combined_X, combined_F, ref_point, max_archive)

            # Update best HV
            current_hv = self._compute_hypervolume(F_archive, ref_point)
            if current_hv > best_hv:
                best_hv = current_hv

        # Final polish: local refinement around best archive points
        if used < self.budget and time.time() < deadline - 10:
            X_archive, F_archive = self._local_refinement(problem, X_archive, F_archive, lo, hi, ref_point, used, deadline, max_archive)
            used = self.budget

        return X_archive

    def _stratified_sample(self, rng, lo, hi, n, dim):
        """Generate stratified samples for better initial spread."""
        points = np.empty((n, dim))
        for d in range(dim):
            # Divide range into n bins and sample one from each
            bins = np.linspace(lo[d], hi[d], n + 1)
            for i in range(n):
                points[i, d] = rng.uniform(bins[i], bins[i + 1])
        return points

    def _select_parent_hype(self, F, ref_point, rng):
        """Select parent based on marginal hypervolume contribution (HypE-style)."""
        n = len(F)
        if n == 0:
            return 0
        if n == 1:
            return 0

        # Compute marginal hypervolume contribution for each point
        # For 2D: contribution of point i is the area it uniquely covers
        # Sort by first objective
        sort_idx = np.argsort(F[:, 0])
        F_sorted = F[sort_idx]

        # Create inverse mapping: position in sorted order -> original index
        inv_sort_idx = np.empty(n, dtype=int)
        inv_sort_idx[sort_idx] = np.arange(n)

        contributions = np.zeros(n)
        for pos in range(n):
            i = inv_sort_idx[pos]
            if pos == 0:
                # Leftmost point: covers from its f1 to next point's f1
                next_f1 = F_sorted[pos + 1, 0] if pos + 1 < n else ref_point[0]
                w = max(0, next_f1 - F_sorted[pos, 0])
                h = max(0, ref_point[1] - F_sorted[pos, 1])
                contributions[i] = w * h
            elif pos == n - 1:
                # Rightmost point
                w = max(0, ref_point[0] - F_sorted[pos, 0])
                h = max(0, ref_point[1] - F_sorted[pos, 1])
                contributions[i] = w * h
            else:
                # Middle point: only contributes if it's not dominated in the sorted chain
                next_f1 = F_sorted[pos + 1, 0]
                prev_f1 = F_sorted[pos - 1, 0]
                w = max(0, next_f1 - prev_f1)
                h_above_prev = max(0, F_sorted[pos - 1, 1] - F_sorted[pos, 1])
                h_below_next = max(0, F_sorted[pos + 1, 1] - F_sorted[pos, 1])
                h = min(h_above_prev, h_below_next)
                contributions[i] = w * h

        # Normalize to probabilities
        total = contributions.sum()
        if total <= 0:
            return rng.integers(n)
        probs = contributions / total
        return rng.choice(n, p=probs)

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
        # Iteratively remove point with smallest marginal contribution
        keep_indices = np.arange(len(F_nd))
        current_F = F_nd.copy()
        current_X = X_nd.copy()

        while len(keep_indices) > max_size:
            # Compute marginal contributions
            n_curr = len(current_F)
            sort_idx = np.argsort(current_F[:, 0])
            F_sorted = current_F[sort_idx]
            inv_sort_idx = np.empty(n_curr, dtype=int)
            inv_sort_idx[sort_idx] = np.arange(n_curr)

            contributions = np.zeros(n_curr)
            for pos in range(n_curr):
                i = inv_sort_idx[pos]
                if pos == 0:
                    next_f1 = F_sorted[pos + 1, 0] if pos + 1 < n_curr else ref_point[0]
                    w = max(0, next_f1 - F_sorted[pos, 0])
                    h = max(0, ref_point[1] - F_sorted[pos, 1])
                    contributions[i] = w * h
                elif pos == n_curr - 1:
                    w = max(0, ref_point[0] - F_sorted[pos, 0])
                    h = max(0, ref_point[1] - F_sorted[pos, 1])
                    contributions[i] = w * h
                else:
                    next_f1 = F_sorted[pos + 1, 0]
                    prev_f1 = F_sorted[pos - 1, 0]
                    w = max(0, next_f1 - prev_f1)
                    h_above_prev = max(0, F_sorted[pos - 1, 1] - F_sorted[pos, 1])
                    h_below_next = max(0, F_sorted[pos + 1, 1] - F_sorted[pos, 1])
                    h = min(h_above_prev, h_below_next)
                    contributions[i] = w * h

            # Remove point with smallest contribution
            min_idx = np.argmin(contributions)
            keep_indices = np.delete(keep_indices, min_idx)
            current_F = current_F[keep_indices]
            current_X = current_X[keep_indices]

        return current_X, current_F

    def _compute_hypervolume(self, F, ref_point):
        """Compute exact hypervolume for 2D objective space."""
        n = len(F)
        if n == 0:
            return 0.0
        
        # Filter to points strictly better than reference point
        mask = np.all(F < ref_point, axis=1)
        F_valid = F[mask]
        if len(F_valid) == 0:
            return 0.0
        
        # Sort by first objective
        sort_idx = np.argsort(F_valid[:, 0])
        F_sorted = F_valid[sort_idx]
        
        hv = 0.0
        prev_f1 = 0.0
        for i in range(len(F_sorted)):
            f1, f2 = F_sorted[i]
            width = f1 - prev_f1
            height = ref_point[1] - f2
            if width > 0 and height > 0:
                hv += width * height
            prev_f1 = f1
        
        return hv

    def _local_refinement(self, problem, X_archive, F_archive, lo, hi, ref_point, used, deadline, max_archive):
        """Local refinement around best archive points to polish the front."""
        start_time = time.time()
        
        # Select top 5 points by hypervolume contribution for refinement
        n = len(X_archive)
        if n == 0:
            return X_archive, F_archive
        
        # Compute contributions to select best points
        sort_idx = np.argsort(F_archive[:, 0])
        F_sorted = F_archive[sort_idx]
        inv_sort_idx = np.empty(n, dtype=int)
        inv_sort_idx[sort_idx] = np.arange(n)
        
        contributions = np.zeros(n)
        for pos in range(n):
            i = inv_sort_idx[pos]
            if pos == 0:
                next_f1 = F_sorted[pos + 1, 0] if pos + 1 < n else ref_point[0]
                w = max(0, next_f1 - F_sorted[pos, 0])
                h = max(0, ref_point[1] - F_sorted[pos, 1])
                contributions[i] = w * h
            elif pos == n - 1:
                w = max(0, ref_point[0] - F_sorted[pos, 0])
                h = max(0, ref_point[1] - F_sorted[pos, 1])
                contributions[i] = w * h
            else:
                next_f1 = F_sorted[pos + 1, 0]
                prev_f1 = F_sorted[pos - 1, 0]
                w = max(0, next_f1 - prev_f1)
                h_above_prev = max(0, F_sorted[pos - 1, 1] - F_sorted[pos, 1])
                h_below_next = max(0, F_sorted[pos + 1, 1] - F_sorted[pos, 1])
                h = min(h_above_prev, h_below_next)
                contributions[i] = w * h
        
        # Select top 5 for refinement
        top_k = min(5, n)
        top_indices = np.argsort(contributions)[-top_k:]
        
        new_X = []
        new_F = []
        rng = np.random.default_rng(123)
        
        for idx in top_indices:
            if time.time() > deadline - 5:
                break
            if used >= self.budget:
                break
            
            parent = X_archive[idx]
            # Small perturbations for local search
            for step in range(3):
                if time.time() > deadline - 5:
                    break
                if used >= self.budget:
                    break
                
                # Very small sigma for local refinement
                sigma = 0.02 * (hi - lo)
                child = parent + rng.normal(0, 1, self.dim) * sigma
                child = np.clip(child, lo, hi)
                
                child_F = problem(child)
                used += 1
                new_X.append(child)
                new_F.append(child_F)
        
        if len(new_X) > 0:
            new_X_arr = np.array(new_X)
            new_F_arr = np.array(new_F)
            combined_X = np.vstack([X_archive, new_X_arr])
            combined_F = np.vstack([F_archive, new_F_arr])
            X_archive, F_archive = self._prune_archive(combined_X, combined_F, ref_point, max_archive)
        
        return X_archive, F_archive
# EVOLVE-BLOCK-END
