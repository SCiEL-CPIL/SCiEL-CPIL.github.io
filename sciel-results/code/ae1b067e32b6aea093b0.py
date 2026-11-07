"""Rewritten optimizer for BBOB f23 (Katsuura) using Adaptive Hypercube Subdivision."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

class Optimizer:
    """Adaptive Hypercube Subdivision for Katsuura on [-5,5]^d."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget
        deadline = time.time() + 280  # Safety margin

        best_x = None
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            calls += 1
            return x, fx

        # Initialize with a coarse grid to ensure global coverage
        # We use 2^d initial boxes
        n_per_dim = 2
        total_initial_boxes = n_per_dim ** d
        
        # If budget is too small for 2^d, reduce n_per_dim
        while n_per_dim ** d > B // 2:
            n_per_dim -= 1
            if n_per_dim < 1:
                n_per_dim = 1
                break
        
        total_initial_boxes = n_per_dim ** d
        
        # Create initial boxes
        # Box is represented by (lower_bound, upper_bound, center_f, center_x, id)
        # We use a heap where the smallest f is at the top
        # To ensure we explore different regions, we might want to limit how deep we go
        # into a single branch before looking at others.
        
        # Strategy: 
        # 1. Generate initial grid points.
        # 2. Evaluate them.
        # 3. Push all initial boxes into a priority queue.
        # 4. Pop the best box. Subdivide it. Push children.
        # 5. To avoid getting stuck in one local minimum, we can use a "depth" limit 
        #    or a "merit" function that penalizes deep boxes if their improvement is slow.
        #    For now, we stick to pure best-first search as per directive, but 
        #    we initialize with a grid to spread out the starts.

        boxes = []
        box_id_counter = 0
        
        # Generate grid coordinates
        grid_coords = np.linspace(-5.0, 5.0, n_per_dim + 1)
        
        for idx in range(total_initial_boxes):
            coords = np.unravel_index(idx, [n_per_dim] * d)
            lower = np.array([grid_coords[c] for c in coords], dtype=np.float64)
            upper = np.array([grid_coords[c + 1] for c in coords], dtype=np.float64)
            center = (lower + upper) / 2.0
            
            cx, cf = safe_call(center)
            if cf < best_f:
                best_f = cf
                best_x = cx.copy()
            
            # Push box: (f_value, id, lower, upper, center, depth)
            # Using id as tie-breaker to ensure deterministic behavior
            heapq.heappush(boxes, (cf, box_id_counter, lower, upper, center, 0))
            box_id_counter += 1

        # Subdivision loop
        # We stop subdividing a box if its size is very small
        min_size = 1e-4
        
        while calls < B - 10 and time.time() < deadline - 5:
            if not boxes:
                break
            
            # Pop the best box
            f_val, bid, lower, upper, center, depth = heapq.heappop(boxes)
            
            # Check if box is too small to subdivide effectively
            side_len = np.max(upper - lower)
            if side_len < min_size:
                continue

            # Subdivide into 2^d children
            # Each dimension is split in half
            # We iterate over all 2^d combinations of lower/upper halves
            
            # To optimize, we can generate the 2^d sub-boxes
            # For d=5, 2^5=32 children.
            
            # We need to be careful with budget.
            # If we don't have enough budget for 32 calls, we might skip subdivision
            # or subdivide only along the most promising axis (Coordinate-wise).
            # Directive says "subdivides ... into 2^d sub-cubes".
            # So we do full bisection.
            
            if calls + (2 ** d) > B:
                # Not enough budget for full subdivision, try to do what we can?
                # Or just stop?
                # Let's try to do coordinate-wise refinement if full bisection is not possible.
                # But for simplicity and adherence to directive, we stop if we can't afford 2^d.
                # However, we can afford 1 call.
                # Let's do a coordinate-wise step for the last few calls.
                pass 
            else:
                # Generate sub-boxes
                # We can use itertools.product or recursive generation
                # For d=5, 32 is small.
                
                # Create list of new boxes
                new_boxes = []
                
                # Helper to generate sub-boxes
                # We split each dimension into [lower, mid] and [mid, upper]
                mids = (lower + upper) / 2.0
                
                # Iterate over all 2^d combinations
                for mask in range(2 ** d):
                    new_lower = lower.copy()
                    new_upper = upper.copy()
                    for i in range(d):
                        if (mask >> i) & 1:
                            new_lower[i] = mids[i]
                        else:
                            new_upper[i] = mids[i]
                    
                    new_center = (new_lower + new_upper) / 2.0
                    ncx, ncf = safe_call(new_center)
                    
                    if ncf < best_f:
                        best_f = ncf
                        best_x = ncx.copy()
                    
                    new_boxes.append((ncf, box_id_counter, new_lower, new_upper, new_center, depth + 1))
                    box_id_counter += 1
                
                # Push all new boxes into the heap
                for box in new_boxes:
                    heapq.heappush(boxes, box)

        # Final Polish: If we have remaining budget, do a coordinate-wise fine tuning
        # from the best point found.
        if best_x is not None and calls < B - 5 and time.time() < deadline - 5:
            step = 0.01
            # Adaptive step reduction
            for _ in range(5): # 5 levels of refinement
                for coord in range(d):
                    if calls >= B - 2 or time.time() > deadline - 2:
                        break
                    # Try positive
                    xp = best_x.copy()
                    xp[coord] += step
                    xp = np.clip(xp, -5.0, 5.0)
                    xp, fp = safe_call(xp)
                    if fp < best_f:
                        best_f = fp
                        best_x = xp
                    else:
                        # Try negative
                        xn = best_x.copy()
                        xn[coord] -= step
                        xn = np.clip(xn, -5.0, 5.0)
                        xn, fn = safe_call(xn)
                        if fn < best_f:
                            best_f = fn
                            best_x = xn
                step *= 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
