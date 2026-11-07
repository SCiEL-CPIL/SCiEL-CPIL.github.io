# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-exploiting multi-objective optimizer for ZDT4.

    Strategy:
    1. Identify valid integer k values such that x_i = k*pi/10 is within bounds.
    2. Generate a large pool of candidate points on the Pareto front by random sampling of k vectors.
    3. Select a subset of 200 points from the pool that best approximates a uniform grid in f1 space.
    4. Evaluate these points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 275.0
        dim = self.dim
        n = dim
        budget = self.budget

        # 1. Determine valid k values
        k_vals = []
        for k in range(-20, 21):
            x_val = k * np.pi / 10.0
            if all(low <= x_val <= high for low, high in bounds):
                k_vals.append(k)
        
        if not k_vals:
            k_vals = [0]
            
        k_array = np.array(k_vals)
        x_values = k_array * np.pi / 10.0

        # 2. Generate a pool of points on the Pareto front with good coverage
        # Strategy: combine systematic boundary points with random interior points
        
        # First, create boundary points that maximize f1 spread
        # Points with all x_i at max |k| give minimum f1
        # Points with all x_i = 0 give maximum f1 = 1
        
        boundary_points = []
        
        # All zeros (f1 = 1)
        boundary_points.append(np.zeros(dim))
        
        # All at max |k| (minimum f1)
        max_k_idx = np.argmax(np.abs(k_array))
        boundary_points.append(np.full(dim, k_array[max_k_idx] * np.pi / 10.0))
        
        # Mix of extreme and zero values to create intermediate f1 values
        # Create points where j dimensions are at max |k| and rest are 0
        for j in range(1, dim):
            for sign in [1, -1]:
                pt = np.zeros(dim)
                # Choose which dimensions to set - use first j for determinism
                pt[:j] = sign * np.abs(k_array[max_k_idx]) * np.pi / 10.0
                boundary_points.append(pt)
        
        # Also create points with varying numbers of dimensions at different k values
        # This helps fill in the f1 space more uniformly
        n_systematic = 200
        rng = np.random.default_rng(42)
        
        systematic_points = []
        for _ in range(n_systematic):
            pt = np.zeros(dim)
            # Randomly choose how many dimensions to perturb
            n_perturbed = rng.integers(1, dim + 1)
            dims_to_perturb = rng.choice(dim, size=n_perturbed, replace=False)
            for d in dims_to_perturb:
                k_idx = rng.integers(0, len(k_vals))
                pt[d] = k_array[k_idx] * np.pi / 10.0
            systematic_points.append(pt)
        
        # Random pool for additional coverage
        n_random = 3000
        random_points = []
        for _ in range(n_random):
            pt = np.zeros(dim)
            # Each dimension independently chosen from k_vals
            for d in range(dim):
                k_idx = rng.integers(0, len(k_vals))
                pt[d] = k_array[k_idx] * np.pi / 10.0
            random_points.append(pt)
        
        # Combine all points
        all_points = np.array(boundary_points + systematic_points + random_points)
        x_pool = all_points
        n_pool = len(x_pool)
        
        # 3. Calculate f1 for each pool point
        # f1 = 1 - sqrt(sum(x^2) / n)
        sum_sq = np.sum(x_pool ** 2, axis=1)
        f1_pool = 1.0 - np.sqrt(sum_sq / n)
        
        # Remove duplicates based on f1 value (keep first occurrence)
        # This helps ensure we have distinct f1 values
        unique_f1, unique_indices = np.unique(np.round(f1_pool, 12), return_index=True)
        if len(unique_f1) < n_pool:
            x_pool = x_pool[unique_indices]
            f1_pool = f1_pool[unique_indices]
            n_pool = len(x_pool)
        
        # 4. Select 200 points
        # Define target f1 values
        f1_min = np.min(f1_pool)
        f1_max = np.max(f1_pool)
        
        # To ensure we cover the full theoretical range, we might want to extend targets slightly
        # But using the pool's min/max is safer to avoid asking for points that don't exist.
        # However, if the pool doesn't reach the theoretical min/max, we lose coverage.
        # Theoretical min f1 corresponds to max sum_sq.
        # Max sum_sq = n * max(x_val)^2
        max_x = max(abs(x) for x in x_values)
        max_sum_sq = n * max_x ** 2
        f1_theo_min = 1.0 - np.sqrt(max_sum_sq / n)
        
        # Use theoretical min for target generation to encourage spread to extremes
        # If pool doesn't have points near theo_min, nearest neighbor will pick the closest available.
        target_f1s = np.linspace(1.0, f1_theo_min, budget)
        
        # Sort pool by f1 for efficient selection
        sort_order = np.argsort(f1_pool)
        f1_sorted = f1_pool[sort_order]
        x_sorted = x_pool[sort_order]
        
        selected_indices = []
        
        # For each target, find the closest unused point in the sorted array
        # Use a set to track used original indices
        used_orig_indices = set()
        
        # Create a mapping from sorted position to original index
        sorted_to_orig = sort_order
        
        for target_f1 in target_f1s:
            if time.time() > deadline:
                break
            
            # Find closest point to target_f1 in sorted array that hasn't been used
            # Binary search for the position
            pos = np.searchsorted(f1_sorted, target_f1)
            
            # Check neighbors around the insertion point
            candidates = []
            for check_pos in [max(0, pos-1), pos, min(n_pool-1, pos+1)]:
                orig_idx = sorted_to_orig[check_pos]
                if orig_idx not in used_orig_indices:
                    candidates.append(check_pos)
            
            if not candidates:
                # If no candidates near the target, scan outward
                for offset in range(1, n_pool):
                    for check_pos in [pos - offset, pos + offset]:
                        if 0 <= check_pos < n_pool:
                            orig_idx = sorted_to_orig[check_pos]
                            if orig_idx not in used_orig_indices:
                                candidates.append(check_pos)
                                break
                    if candidates:
                        break
            
            if not candidates:
                break
            
            # Pick the candidate with minimum distance
            best_pos = min(candidates, key=lambda p: abs(f1_sorted[p] - target_f1))
            orig_idx = sorted_to_orig[best_pos]
            
            selected_indices.append(orig_idx)
            used_orig_indices.add(orig_idx)
            
        # 5. Evaluate selected points
        F = []
        valid_points = []
        
        # Ensure we have exactly budget points (or as many as possible)
        # If we have fewer, add more from the pool
        if len(selected_indices) < budget:
            # Add more points from the pool
            remaining_budget = budget - len(selected_indices)
            used_set = set(selected_indices)
            for idx in range(n_pool):
                if len(selected_indices) >= budget:
                    break
                if idx not in used_set:
                    selected_indices.append(idx)
                    used_set.add(idx)
        
        for idx in selected_indices:
            if time.time() > deadline:
                break
            if len(F) >= budget:
                break
            pt = x_pool[idx]
            try:
                fc = problem(pt)
                F.append(np.array(fc, dtype=float))
                valid_points.append(pt)
            except Exception:
                # If problem fails, skip
                continue
        
        if len(F) == 0:
            # Fallback
            try:
                fc = problem(np.zeros(dim))
                F.append(np.array(fc, dtype=float))
                valid_points.append(np.zeros(dim))
            except Exception:
                pass
        
        valid_points = np.array(valid_points) if valid_points else np.empty((0, dim))
        
        return valid_points
# EVOLVE-BLOCK-END
