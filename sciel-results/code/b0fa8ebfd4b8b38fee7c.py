import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin from 330s limit
        
        dim = self.dim
        budget = self.budget
        calls_used = 0
        
        def check_time():
            return time.time() > deadline - 5

        # Helper to evaluate and track calls
        def eval_func(x):
            nonlocal calls_used
            if check_time():
                return float('inf')
            calls_used += 1
            if calls_used > budget:
                return float('inf')
            return func(x)

        # Phase 1: Diverse Initialization using Halton Sequence
        # Halton sequence provides low-discrepancy points, good for covering the space
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        n_initial = 25
        points = []
        values = []
        
        # Generate Halton points in [0, 1]^dim
        for i in range(1, n_initial + 1):
            x = np.array([halton(i, p) for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:dim]])
            # Map to [-5, 5]^dim
            x = x * 10.0 - 5.0
            f = eval_func(x)
            points.append(x)
            values.append(f)
            
        if check_time():
            best_idx = np.argmin(values)
            return points[best_idx], values[best_idx]

        # Sort points by value
        sorted_indices = np.argsort(values)
        top_k = 3
        top_indices = sorted_indices[:top_k]
        
        best_x = points[top_indices[0]].copy()
        best_f = values[top_indices[0]]
        
        # Phase 2: Adaptive Grid Refinement for top K points
        # We will allocate remaining budget to the top K points
        remaining_budget = budget - calls_used
        budget_per_point = remaining_budget // top_k
        
        for idx in top_indices:
            if check_time() or calls_used >= budget:
                break
                
            x0 = points[idx].copy()
            f0 = values[idx]
            
            # Initialize local box around x0
            # Box size 1.0 in each dimension
            box_lo = x0 - 0.5
            box_hi = x0 + 0.5
            
            # Ensure box is within [-5, 5]
            box_lo = np.clip(box_lo, -5, 5)
            box_hi = np.clip(box_hi, -5, 5)
            
            # If box is degenerate, skip
            if np.all(box_hi - box_lo < 1e-6):
                continue
                
            local_calls = 0
            local_budget = budget_per_point
            max_iter = 20 # Safety cap on iterations
            
            for _ in range(max_iter):
                if check_time() or calls_used >= budget or local_calls >= local_budget:
                    break
                
                # Generate 2^dim sub-box centers
                # This is 32 points in 5D
                n_sub = 2 ** dim
                sub_centers = np.zeros((n_sub, dim))
                
                for i in range(n_sub):
                    # Decode binary representation of i to determine sub-box
                    for j in range(dim):
                        # If bit j of i is 1, take upper half, else lower half
                        if (i >> j) & 1:
                            sub_centers[i, j] = (box_lo[j] + box_hi[j]) / 2.0 + (box_hi[j] - box_lo[j]) / 4.0
                        else:
                            sub_centers[i, j] = (box_lo[j] + box_hi[j]) / 2.0 - (box_hi[j] - box_lo[j]) / 4.0
                
                # Evaluate all sub-box centers
                sub_values = []
                for x in sub_centers:
                    f = eval_func(x)
                    sub_values.append(f)
                    local_calls += 1
                    
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        
                if check_time() or calls_used >= budget or local_calls >= local_budget:
                    break
                
                # Find the best sub-box
                best_sub_idx = np.argmin(sub_values)
                
                # Update the box to be the best sub-box
                for j in range(dim):
                    if (best_sub_idx >> j) & 1:
                        # Upper half
                        box_lo[j] = (box_lo[j] + box_hi[j]) / 2.0
                        box_hi[j] = box_hi[j]
                    else:
                        # Lower half
                        box_lo[j] = box_lo[j]
                        box_hi[j] = (box_lo[j] + box_hi[j]) / 2.0
                
                # Check if box size is too small
                if np.all(box_hi - box_lo < 1e-5):
                    break

        return best_x, best_f
