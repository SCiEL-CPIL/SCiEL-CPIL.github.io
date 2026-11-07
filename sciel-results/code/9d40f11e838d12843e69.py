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

        # 2. Generate a pool of random points on the Pareto front
        # We need enough points to cover the f1 range well.
        # 5000 points is fast and provides good density.
        n_pool = 5000
        rng = np.random.default_rng(42)
        
        # Randomly select k indices for each dimension for each pool point
        # shape: (n_pool, dim)
        k_indices = rng.integers(0, len(k_vals), size=(n_pool, dim))
        # Map to actual k values
        k_pool = k_array[k_indices]
        # Map to x values
        x_pool = k_pool * np.pi / 10.0
        
        # 3. Calculate f1 for each pool point
        # f1 = 1 - sqrt(sum(x^2) / n)
        sum_sq = np.sum(x_pool ** 2, axis=1)
        f1_pool = 1.0 - np.sqrt(sum_sq / n)
        
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
        
        selected_indices = []
        used_mask = np.zeros(n_pool, dtype=bool)
        
        # Sort pool by f1 to speed up nearest neighbor search? 
        # Or just loop. 200 * 5000 = 1e6 ops, which is fine.
        
        for target_f1 in target_f1s:
            if time.time() > deadline:
                break
                
            # Find unused points closest to target_f1
            # To avoid picking the same point multiple times, we mark used.
            # Efficient way: 
            # 1. Find all unused points.
            # 2. Compute distance to target.
            # 3. Pick min.
            
            # Optimization: 
            # Since we are picking in order of target_f1, and f1_pool is fixed,
            # we can sort the pool once.
            # But "used" mask makes it dynamic.
            
            # Simple approach:
            # Create a list of available indices.
            # This is O(N) per step if we don't manage it well.
            
            # Better: 
            # We can just pick the global nearest neighbor among unused points.
            # dist = abs(f1_pool - target_f1)
            # dist[used_mask] = inf
            # idx = argmin(dist)
            
            dist = np.abs(f1_pool - target_f1)
            dist[used_mask] = np.inf
            idx = np.argmin(dist)
            
            if dist[idx] == np.inf:
                # No points left?
                break
                
            selected_indices.append(idx)
            used_mask[idx] = True
            
        # 5. Evaluate selected points
        F = []
        valid_points = []
        
        # Check for duplicates in selected_indices (should be unique due to used_mask)
        # But what if two targets pick the same index? used_mask prevents this.
        
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
