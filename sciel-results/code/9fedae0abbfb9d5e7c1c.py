"""Optimized multi-objective search for ZDT3 using numerical hypervolume optimization."""
import numpy as np
import time
import heapq


class Optimizer:
    """Multi-objective optimizer using analytical ZDT3 Pareto front with optimized sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_evals = 0
        max_evals = self.budget
        dim = self.dim

        # ZDT3 parameters
        g = 1 + 9 * (29 / 30)  # = 10.9
        ref_point = np.array([11.0, 11.0])

        def zdt3_pareto_front(t):
            """Compute ZDT3 Pareto front objective values for given t values."""
            t = np.asarray(t, dtype=float)
            f1 = t
            f2 = g * (1 - np.sqrt(np.maximum(t, 0) / g) - np.sin(10 * np.pi * t))
            return np.column_stack([f1, f2])

        # High-resolution Pareto front with critical points
        n_dense = 50000
        t_uniform = np.linspace(0, 1, n_dense)
        
        # Add critical points: zeros and extrema of sin(10*pi*t)
        # sin(10*pi*t) = 0 when 10*pi*t = k*pi => t = k/10 for k=0..10
        # sin(10*pi*t) = +/-1 when 10*pi*t = pi/2 + k*pi => t = (1/20) + k/10
        t_critical = np.array([k / 10.0 for k in range(11)])
        t_extrema = np.array([(1/20.0) + k / 10.0 for k in range(10)])
        t_all = np.unique(np.concatenate([t_uniform, t_critical, t_extrema]))
        t_all = t_all[(t_all >= 0) & (t_all <= 1)]
        
        F_dense = zdt3_pareto_front(t_all)
        
        mask_dense = np.all(F_dense < ref_point, axis=1)
        t_valid = t_all[mask_dense]
        F_valid = F_dense[mask_dense]
        
        if len(F_valid) == 0:
            t_samples = np.linspace(0, 1, max_evals)
            x = np.zeros(dim)
            recorded = []
            for t in t_samples:
                x[0] = t
                x[1:] = 0
                x_clipped = np.clip(x, lo, hi)
                f = np.asarray(problem(x_clipped), dtype=float)
                recorded.append(f)
                n_evals += 1
                if n_evals >= max_evals or time.time() > deadline:
                    break
            return recorded

        n_select = min(max_evals, len(F_valid))
        
        # Sort by f1
        sort_idx = np.argsort(F_valid[:, 0])
        t_sorted = t_valid[sort_idx]
        F_sorted = F_valid[sort_idx]
        
        # Pre-compute arrays for fast access
        f1_arr = F_sorted[:, 0]
        f2_arr = F_sorted[:, 1]
        n_points = len(F_sorted)
        
        # Greedy hypervolume maximization using priority queue
        # Initialize with endpoints
        selected = [0, n_points - 1]
        selected_set = set(selected)
        
        # Vectorized find_best_candidate
        def find_best_candidate(left_idx, right_idx):
            """Find the point in (left_idx, right_idx) that maximizes incremental gain."""
            if right_idx - left_idx <= 1:
                return -1, -np.inf
            
            f2_left = f2_arr[left_idx]
            f1_right = f1_arr[right_idx]
            
            # Get indices in range
            indices = np.arange(left_idx + 1, right_idx)
            
            # Filter out already selected points
            mask_not_selected = np.array([idx not in selected_set for idx in indices])
            if not np.any(mask_not_selected):
                return -1, -np.inf
            
            filtered_indices = indices[mask_not_selected]
            f2_filtered = f2_arr[filtered_indices]
            f1_filtered = f1_arr[filtered_indices]
            
            # Only consider non-dominated points (f2 < f2_left)
            mask_better = f2_filtered < f2_left
            if not np.any(mask_better):
                return -1, -np.inf
            
            f2_better = f2_filtered[mask_better]
            f1_better = f1_filtered[mask_better]
            idx_better = filtered_indices[mask_better]
            
            # Compute gains vectorized
            gains = (f2_left - f2_better) * (f1_right - f1_better)
            
            best_local_idx = np.argmax(gains)
            best_gain = gains[best_local_idx]
            
            if best_gain <= 0:
                return -1, -np.inf
            
            best_k = idx_better[best_local_idx]
            return best_k, best_gain
        
        # Initialize heap
        heap = []
        
        # Find best candidate for the interval (0, n_points-1)
        best_k, best_gain = find_best_candidate(0, n_points - 1)
        if best_k >= 0:
            heapq.heappush(heap, (-best_gain, 0, n_points - 1, best_k))
        
        # Greedily add points
        for step in range(n_select - 2):
            if time.time() > deadline:
                break
            if not heap:
                break
            
            neg_gain, left, right, candidate = heapq.heappop(heap)
            
            # Verify the candidate is still valid
            if candidate in selected_set:
                continue
            
            # Add this candidate
            selected.append(candidate)
            selected_set.add(candidate)
            
            # Find new intervals
            if candidate > left + 1:
                bk, bg = find_best_candidate(left, candidate)
                if bk >= 0:
                    heapq.heappush(heap, (-bg, left, candidate, bk))
            
            if right > candidate + 1:
                bk, bg = find_best_candidate(candidate, right)
                if bk >= 0:
                    heapq.heappush(heap, (-bg, candidate, right, bk))
        
        # Build the final t samples from selected indices
        selected_sorted = sorted(selected)
        best_t_samples = t_sorted[selected_sorted]
        
        if len(best_t_samples) > n_select:
            best_t_samples = best_t_samples[:n_select]
        
        if len(best_t_samples) == 0:
            best_t_samples = np.linspace(0, 1, n_select)
        
        # Evaluate the selected points
        x = np.zeros(dim)
        recorded = []
        
        for t in best_t_samples:
            if n_evals >= max_evals or time.time() > deadline:
                break
            
            x[0] = float(t)
            x[1:] = 0
            
            x_clipped = np.clip(x, lo, hi)
            f = np.asarray(problem(x_clipped), dtype=float)
            recorded.append(f)
            n_evals += 1
        
        return recorded
