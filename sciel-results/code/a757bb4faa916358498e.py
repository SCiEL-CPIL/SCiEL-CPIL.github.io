import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        ref_point = np.array([1.1, 1.3, 1.1])
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate Halton sequence candidates
        n_candidates = 300
        candidates = self._halton_sequence(n_candidates, self.dim)
        candidates = lo + (hi - lo) * candidates
        
        archive = []
        archive_objs = []
        evaluated_indices = set()
        X_train = []
        F_train = []
        
        # Initialize with 10 random points
        init_count = 10
        rng = np.random.default_rng(42)
        init_indices = rng.choice(n_candidates, size=init_count, replace=False)
        
        for i in init_indices:
            x = candidates[i]
            f = problem(x)
            evaluated_indices.add(i)
            X_train.append(x)
            F_train.append(f)
            self._update_archive(x, f, archive, archive_objs)
            
        X_train = np.array(X_train)
        F_train = np.array(F_train)
        
        # Greedy loop
        for step in range(init_count, self.budget):
            remaining = [i for i in range(n_candidates) if i not in evaluated_indices]
            if not remaining:
                break
            
            best_idx = -1
            best_marginal = -1
            best_point = None
            best_obj = None
            
            # Predict objectives for remaining candidates
            # Use KNN
            k = min(10, len(X_train))
            dists = np.linalg.norm(X_train[None, :, :] - candidates[remaining, None, :], axis=2)
            # dists shape: (len(remaining), len(X_train))
            nn_indices = np.argpartition(dists, k, axis=1)[:, :k]
            # Get weights
            nn_dists = np.take_along_axis(dists, nn_indices, axis=1)
            weights = 1.0 / (nn_dists + 1e-6)
            weights /= np.sum(weights, axis=1, keepdims=True)
            
            # Predict
            # F_train shape: (N, 3)
            # We need to compute weighted sum for each candidate
            # Expand weights to (len(remaining), k, 1)
            w_expanded = weights[:, :, np.newaxis]
            # Get neighbors' objectives
            # nn_indices shape: (len(remaining), k)
            # F_train[nn_indices] shape: (len(remaining), k, 3)
            preds = np.sum(w_expanded * F_train[nn_indices], axis=1)
            
            # Filter out dominated predictions
            # Check dominance against archive
            valid_mask = np.ones(len(remaining), dtype=bool)
            if archive_objs:
                # archive_objs is list of arrays
                # Convert to array for vectorized check
                # This is O(len(remaining) * len(archive))
                # We can do it in a loop for simplicity if archive is small
                for j, af in enumerate(archive_objs):
                    # Check if preds[i] is dominated by af
                    # Dominated if af <= preds[i] and af < preds[i]
                    # Vectorized:
                    dominated = np.all(af <= preds, axis=1) & np.any(af < preds, axis=1)
                    valid_mask &= ~dominated
            
            # Calculate marginal HV for valid candidates
            # To save time, only calculate for a subset if too many?
            # With 300 candidates and small archive, it should be fine.
            
            for j, idx in enumerate(remaining):
                if not valid_mask[j]:
                    continue
                
                f_pred = preds[j]
                
                # Calculate marginal HV
                # Add f_pred to archive_objs
                new_archive_objs = archive_objs + [f_pred]
                new_hv = self._hypervolume_3d(new_archive_objs, ref_point)
                current_hv = self._hypervolume_3d(archive_objs, ref_point) # Cache this?
                
                # Optimization: Calculate current_hv once outside the loop
                # Let's restructure to calculate current_hv once.
                
                # This inner loop is slow. Let's optimize.
                pass
        
        # The above loop is inefficient. Let's rewrite the greedy step.
        
        # Reset state for a cleaner implementation
        archive = []
        archive_objs = []
        evaluated_indices = set()
        X_train = []
        F_train = []
        
        init_count = 10
        rng = np.random.default_rng(42)
        init_indices = rng.choice(n_candidates, size=init_count, replace=False)
        
        for i in init_indices:
            x = candidates[i]
            f = problem(x)
            evaluated_indices.add(i)
            X_train.append(x)
            F_train.append(f)
            self._update_archive(x, f, archive, archive_objs)
            
        X_train = np.array(X_train)
        F_train = np.array(F_train)
        
        for step in range(init_count, self.budget):
            remaining = [i for i in range(n_candidates) if i not in evaluated_indices]
            if not remaining:
                break
            
            current_hv = self._hypervolume_3d(archive_objs, ref_point)
            
            # Predict
            k = min(10, len(X_train))
            # Vectorized KNN prediction
            dists = np.linalg.norm(X_train[None, :, :] - candidates[remaining, None, :], axis=2)
            nn_indices = np.argpartition(dists, k, axis=1)[:, :k]
            nn_dists = np.take_along_axis(dists, nn_indices, axis=1)
            weights = 1.0 / (nn_dists + 1e-6)
            weights /= np.sum(weights, axis=1, keepdims=True)
            preds = np.sum(weights[:, :, np.newaxis] * F_train[nn_indices], axis=1)
            
            # Filter dominated
            valid_mask = np.ones(len(remaining), dtype=bool)
            if archive_objs:
                for af in archive_objs:
                    dominated = np.all(af <= preds, axis=1) & np.any(af < preds, axis=1)
                    valid_mask &= ~dominated
            
            # Find best marginal HV
            best_marginal = -1
            best_local_idx = -1
            
            # Iterate only over valid candidates
            valid_indices = np.where(valid_mask)[0]
            
            for j in valid_indices:
                f_pred = preds[j]
                new_hv = self._hypervolume_3d(archive_objs + [f_pred], ref_point)
                marginal = new_hv - current_hv
                
                if marginal > best_marginal:
                    best_marginal = marginal
                    best_local_idx = j
            
            if best_local_idx == -1:
                # Fallback: pick random
                idx = remaining[rng.integers(len(remaining))]
                x = candidates[idx]
                f = problem(x)
                evaluated_indices.add(idx)
                X_train = np.vstack([X_train, x])
                F_train = np.vstack([F_train, f])
                self._update_archive(x, f, archive, archive_objs)
            else:
                idx = remaining[best_local_idx]
                x = candidates[idx]
                f = problem(x)
                evaluated_indices.add(idx)
                X_train = np.vstack([X_train, x])
                F_train = np.vstack([F_train, f])
                self._update_archive(x, f, archive, archive_objs)
        
        return archive

    def _halton_sequence(self, n, dim):
        # Simple Halton sequence implementation
        halton = np.zeros((n, dim))
        for d in range(dim):
            base = d + 2
            h = 0
            f = 1
            for i in range(n):
                f = f / base
                h = h + f * (i // f) % 1
                halton[i, d] = h
        return halton

    def _update_archive(self, x, f, archive, archive_objs):
        # Check if dominated
        for af in archive_objs:
            if np.all(af <= f) and np.any(af < f):
                return
        
        # Remove dominated
        new_archive = []
        new_archive_objs = []
        for p, af in zip(archive, archive_objs):
            if not (np.all(f <= af) and np.any(f < af)):
                new_archive.append(p)
                new_archive_objs.append(af)
        
        new_archive.append(x)
        new_archive_objs.append(f)
        
        # Update in place
        archive.clear()
        archive.extend(new_archive)
        archive_objs.clear()
        archive_objs.extend(new_archive_objs)

    def _hypervolume_3d(self, points, ref):
        if not points:
            return 0.0
        
        pts = np.array(points)
        # Filter
        pts = pts[np.all(pts < ref, axis=1)]
        if len(pts) == 0:
            return 0.0
            
        # Sort by f0
        pts = pts[pts[:, 0].argsort()]
        
        total_vol = 0.0
        prev_f0 = pts[0, 0]
        
        # We need to calculate 2D HV for prefixes
        # To optimize, we can maintain the 2D HV incrementally?
        # No, the set of points changes (grows).
        # We can just call _hypervolume_2d for each prefix.
        
        n = len(pts)
        # Pre-compute 2D HV for all prefixes?
        # _hypervolume_2d is O(M log M).
        # Total O(N^2 log N).
        
        for i in range(n):
            # The slice is from prev_f0 to pts[i, 0]
            # The active set for this slice is pts[0:i] (points with f0 < pts[i, 0])
            # Wait, the slice is [prev_f0, pts[i, 0]].
            # The active points are those with f0 <= prev_f0?
            # No, for any z in [prev_f0, pts[i, 0]], the active points are those with f0 <= z.
            # Since z >= prev_f0, the set of points with f0 <= prev_f0 is a subset.
            # Points with f0 in (prev_f0, z] are also active.
            # So the area is not constant over the slice?
            # Yes, it changes at the f0 values of the points.
            # But we are iterating over the points' f0 values.
            # So the "next" point is pts[i].
            # The slice is [pts[i-1, 0], pts[i, 0]] (with pts[-1, 0] = -inf or min).
            # In this slice, the active set is {p | p[0] <= pts[i-1, 0]}.
            # So we need the 2D HV of the prefix pts[0:i].
            
            if i == 0:
                # Slice from min to pts[0, 0]?
                # Actually, the volume is from -inf to ref.
                # But points are >= 0.
                # Let's assume the slice starts from 0 or min bound.
                # The problem states bounds [0,1].
                # So we can start from 0.
                pass
            
            # Let's use the suffix method which is often cleaner:
            # Sort by f0 DESCENDING.
            # p0, p1, ... pN-1 (p0 has highest f0).
            # HV = sum (f0_{i-1} - f0_i) * Area_2D({p_i, ..., p_N-1})
            # Where f0_{-1} = ref[0].
            
            pass
        
        # Let's implement the Descending Suffix Method
        pts_desc = pts[::-1] # Sorted by f0 descending
        n = len(pts_desc)
        
        # Pre-compute 2D HV for suffixes?
        # Suffix i is pts_desc[i:n]
        # We need Area_2D of pts_desc[i:n]
        
        total_vol = 0.0
        prev_f0 = ref[0]
        
        # We need 2D HV of the suffix.
        # We can compute this on the fly.
        
        for i in range(n):
            current_f0 = pts_desc[i, 0]
            width = prev_f0 - current_f0
            if width <= 0:
                prev_f0 = current_f0
                continue
                
            # Get suffix points (f1, f2)
            suffix_pts = pts_desc[i:, 1:3]
            area = self._hypervolume_2d(suffix_pts, ref[1:3])
            
            total_vol += width * area
            prev_f0 = current_f0
            
        return total_vol

    def _hypervolume_2d(self, points, ref):
        if len(points) == 0:
            return 0.0
            
        pts = np.array(points)
        pts = pts[np.all(pts < ref, axis=1)]
        if len(pts) == 0:
            return 0.0
            
        # Sort by f1 ascending
        pts = pts[pts[:, 0].argsort()]
        
        area = 0.0
        min_f2 = ref[1]
        
        # Sweep from right to left (descending f1)
        # Or left to right?
        # Standard 2D HV:
        # Sort by f1 ascending.
        # Area = sum (v_{j+1} - v_j) * (ref[1] - min_{k<=j} f_{2,k})
        # This assumes the "active" set for the strip [v_j, v_{j+1}] is {p_k | k <= j}.
        # Is this correct?
        # For f1 in [v_j, v_{j+1}), the active points are those with p1 <= f1.
        # Since f1 < v_{j+1}, the points with p1 <= v_j are definitely active.
        # Points with p1 in (v_j, f1] are also active.
        # But v_j are the sorted unique values.
        # If there are no points with p1 in (v_j, v_{j+1}), then the set is constant.
        # Yes, because v_j are the coordinates of the points.
        
        # So we can iterate j from 0 to N-1.
        # Strip j is [v_j, v_{j+1}] (with v_N = ref[0]).
        # Active set is {p_0, ..., p_j}.
        # Min f2 is min(f2_0, ..., f2_j).
        
        n = len(pts)
        min_f2 = ref[1]
        
        for j in range(n):
            # Update min_f2
            if pts[j, 1] < min_f2:
                min_f2 = pts[j, 1]
                
            # Calculate width
            v_j = pts[j, 0]
            v_next = pts[j+1, 0] if j+1 < n else ref[0]
            width = v_next - v_j
            
            if width > 0:
                height = ref[1] - min_f2
                area += width * height
                
        return area
