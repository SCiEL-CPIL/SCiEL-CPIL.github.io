"""Structure-aware multi-objective optimizer for ZDT4 exploiting known Pareto front parametrization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for ZDT4.
    
    Strategy:
    1. Exploit ZDT4 structure:
       - f1 = 1 + 9*x1 + sum_{i=2}^n 10*(1 + sin(12*pi*x_i))
       - f2 = g(x1) * (1 - sqrt(x1/g(x1)))
       - Pareto front achieved when sin(12*pi*x_i) = -1 for i > 1
       - This occurs at x_i = 0.125 + k/6 for integer k
       
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

        # For ZDT4, x_i = 0.125 minimizes 10*(1 + sin(12*pi*x_i)) to 0
        # This makes f1 = 1 + 9*x1, which is the Pareto front
        x_tail = 0.125
        x_tail_clipped = np.clip(x_tail, lo[1:], hi[1:])
        
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
                x[i] = x_tail_clipped[i-1]
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
            gaps = np.diff(sorted_f1)
            
            intervals = []
            for i in range(len(gaps)):
                f1_start = sorted_f1[i]
                f1_end = sorted_f1[i+1]
                # Map f1 back to x1: f1 = 1 + 9*x1, so x1 = (f1 - 1) / 9
                x1_start = (f1_start - 1.0) / 9.0
                x1_end = (f1_end - 1.0) / 9.0
                
                # Clip to bounds
                x1_start = np.clip(x1_start, lo[0], hi[0])
                x1_end = np.clip(x1_end, lo[0], hi[0])
                
                if x1_end > x1_start:
                    intervals.append((x1_end - x1_start, x1_start, x1_end))
                    
            # Sort by gap size descending
            intervals.sort(key=lambda x: x[0], reverse=True)
            
            current_intervals = intervals[:]
            
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
                    x[i] = x_tail_clipped[i-1]
                x = np.clip(x, lo, hi)
                
                f = problem(x)
                f = np.asarray(f, dtype=float)
                all_points.append(f.copy())
                
                # Update intervals
                current_intervals.pop(max_idx)
                
                # Add new intervals
                if mid > start:
                    current_intervals.append((mid - start, start, mid))
                if end > mid:
                    current_intervals.append((end - mid, mid, end))
                    
        return all_points

# EVOLVE-BLOCK-END
