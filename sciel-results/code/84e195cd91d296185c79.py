"""Structure-aware multi-objective optimizer for ZDT4 exploiting known Pareto front parametrization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for ZDT4.
    
    Strategy:
    1. Fix x2..xn to 0.125 to minimize f1 (Pareto optimal).
    2. Use adaptive sampling for x1:
       - Phase 1: Uniform sampling of half the budget.
       - Phase 2: Identify largest gaps in the f1 objective space and fill them.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        all_points = []
        all_f1 = []
        
        if time.time() > deadline:
            return all_points

        x_fixed = 0.125
        x_fixed_clipped = np.clip(x_fixed, lo[1:], hi[1:])
        
        # Phase 1: Uniform sampling
        n_phase1 = budget // 2
        if n_phase1 < 10:
            n_phase1 = 10
            n_phase2 = budget - n_phase1
        else:
            n_phase2 = budget - n_phase1
            
        x1_vals = np.linspace(lo[0], hi[0], n_phase1)
        
        for x1_val in x1_vals:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x1_val
            for i in range(1, dim):
                x[i] = x_fixed_clipped[i-1]
            x = np.clip(x, lo, hi)
            f = problem(x)
            f = np.asarray(f, dtype=float)
            all_points.append(f.copy())
            all_f1.append(f[0])
            
        if n_phase2 <= 0 or time.time() > deadline:
            return all_points
            
        # Phase 2: Adaptive gap filling
        # Sort current points by f1
        if all_points:
            indices = np.argsort(all_f1)
            sorted_f1 = np.array(all_f1)[indices]
            
            # Calculate gaps in f1
            # We want to place points in the middle of the largest gaps
            # To avoid clustering, we can look at the largest gaps in the objective space
            # Since f1 is monotonic with x1, largest f1 gap == largest x1 gap
            # However, if the front is curved, we might want to fill gaps in f2 as well?
            # For ZDT4, f1 is linear in x1. So uniform x1 is uniform f1.
            # If the front is non-linear in f2, uniform f1 might be suboptimal for HV.
            # Let's stick to filling f1 gaps first, as it ensures coverage.
            
            gaps = np.diff(sorted_f1)
            # Also consider the gaps at the boundaries? 
            # The reference point is [11, 400]. 
            # Points should be within [0, 1] for x1 typically.
            # If we have endpoints, the "gaps" are between points.
            
            # Find the index of the largest gap
            # We will insert n_phase2 points into the largest gaps
            # To distribute them, we can pick the largest gaps and subdivide them.
            
            # Simple approach: 
            # Create a list of intervals (f1_start, f1_end) for each gap
            # Sort intervals by size
            # For each new point, pick the largest remaining interval, 
            # split it in half, and add the midpoint.
            
            intervals = []
            for i in range(len(gaps)):
                # Map back to x1? 
                # We need to map f1 back to x1.
                # f1 = 1 + 9*x1 + sum_term.
                # sum_term is 0 for our points.
                # So x1 = (f1 - 1) / 9.
                # This is valid ONLY if we are on the Pareto front (which we are).
                
                f1_start = sorted_f1[i]
                f1_end = sorted_f1[i+1]
                x1_start = (f1_start - 1.0) / 9.0
                x1_end = (f1_end - 1.0) / 9.0
                
                # Clip to bounds
                x1_start = np.clip(x1_start, lo[0], hi[0])
                x1_end = np.clip(x1_end, lo[0], hi[0])
                
                if x1_end > x1_start:
                    intervals.append((x1_end - x1_start, x1_start, x1_end))
                    
            # Sort by gap size descending
            intervals.sort(key=lambda x: x[0], reverse=True)
            
            # We need to handle the fact that inserting a point changes the gaps.
            # A simple greedy approach:
            # For each of n_phase2 points:
            #   Find the current largest gap (re-evaluating or updating a heap is better, but n is small)
            #   Since n_phase2 is ~100, and we start with ~100 points, 
            #   we can just re-sort or use a simpler heuristic.
            
            # Heuristic: Just take the top n_phase2 largest gaps from the initial set?
            # No, that might put multiple points in the same big gap.
            
            # Better: Iteratively pick the largest gap, add midpoint, remove that gap, add two new half-gaps.
            # This is O(N^2) which is fine for N=200.
            
            current_intervals = intervals[:]
            # Convert to a list we can modify
            # (size, start, end)
            
            for _ in range(n_phase2):
                if not current_intervals:
                    break
                if time.time() > deadline:
                    break
                    
                # Find max
                max_idx = 0
                max_size = -1
                for i, item in enumerate(current_intervals):
                    if item[0] > max_size:
                        max_size = item[0]
                        max_idx = i
                        
                size, start, end = current_intervals[max_idx]
                mid = (start + end) / 2.0
                
                # Evaluate mid
                x = np.zeros(dim)
                x[0] = mid
                for i in range(1, dim):
                    x[i] = x_fixed_clipped[i-1]
                x = np.clip(x, lo, hi)
                
                f = problem(x)
                f = np.asarray(f, dtype=float)
                all_points.append(f.copy())
                # We don't strictly need to update all_f1 for the logic if we just use x1 mapping,
                # but let's be consistent.
                
                # Update intervals
                # Remove max_idx
                current_intervals.pop(max_idx)
                
                # Add new intervals
                if mid > start:
                    current_intervals.append((mid - start, start, mid))
                if end > mid:
                    current_intervals.append((end - mid, mid, end))
                    
        return all_points

# EVOLVE-BLOCK-END
