# EVOLVE-BLOCK-START
import numpy as np
import random

def squares_in_square10() -> tuple:
    """
    Maximize sum of side lengths of 10 axis-aligned squares in unit square.
    Uses Multi-start Local Search with Greedy Packing initialization.
    """
    n = 10
    tol = 1e-7

    def is_valid(s, cx, cy):
        """Check if the configuration is valid."""
        for i in range(n):
            if s[i] <= 0:
                return False
            if cx[i] - s[i]/2 < -tol or cx[i] + s[i]/2 > 1 + tol:
                return False
            if cy[i] - s[i]/2 < -tol or cy[i] + s[i]/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2 - tol
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                if dx < t and dy < t:
                    return False
        return True

    def greedy_pack(sizes):
        """
        Attempt to pack squares with given sizes into the unit square.
        Returns (s, cx, cy) if successful, else None.
        Uses a simple shelf packing heuristic.
        """
        s = np.zeros(n)
        cx = np.zeros(n)
        cy = np.zeros(n)
        
        # Sort indices by size descending for better packing
        order = np.argsort(sizes)[::-1]
        
        # Shelf packing
        # We maintain a list of "shelves" (y-intervals) and current x-position for each shelf.
        # For simplicity, we just try to place in a grid-like manner or use a simple recursive fit.
        # A simpler robust heuristic: Try to place each square in the bottom-most, left-most valid position.
        
        # To make it diverse, we can shuffle the order slightly or use different strategies.
        # Here we use a deterministic "bottom-left" fit for the given size order.
        
        # We need to track occupied rectangles to find free space.
        # Since n=10 is small, we can just check against all previously placed squares.
        
        placed = []
        
        for idx in order:
            size = sizes[idx]
            if size > 1.0:
                return None
            
            found = False
            # Candidate positions:
            # 1. Corners: (0,0), (1,0), (0,1), (1,1) adjusted by size/2
            # 2. Adjacent to existing squares: right, top, left, bottom
            
            candidates = []
            
            # Base corners
            candidates.append((size/2, size/2)) # Bottom-Left
            candidates.append((1-size/2, size/2)) # Bottom-Right
            candidates.append((size/2, 1-size/2)) # Top-Left
            candidates.append((1-size/2, 1-size/2)) # Top-Right
            
            # Generate candidates adjacent to placed squares
            for ps, pcx, pcy in placed:
                # Right of ps
                candidates.append((pcx + (ps + size)/2, pcy))
                # Left of ps
                candidates.append((pcx - (ps + size)/2, pcy))
                # Top of ps
                candidates.append((pcx, pcy + (ps + size)/2))
                # Bottom of ps
                candidates.append((pcx, pcy - (ps + size)/2))
                
                # Also try corners of the bounding box of placed squares?
                # For now, stick to immediate neighbors.
            
            # Check candidates
            valid_pos = None
            for px, py in candidates:
                # Check bounds
                if px - size/2 < -tol or px + size/2 > 1 + tol:
                    continue
                if py - size/2 < -tol or py + size/2 > 1 + tol:
                    continue
                
                # Check overlap with placed
                overlap = False
                for ps, pcx, pcy in placed:
                    t = (size + ps) / 2 - tol
                    if abs(px - pcx) < t and abs(py - pcy) < t:
                        overlap = True
                        break
                
                if not overlap:
                    valid_pos = (px, py)
                    break
            
            if valid_pos:
                s[idx] = size
                cx[idx] = valid_pos[0]
                cy[idx] = valid_pos[1]
                placed.append((size, valid_pos[0], valid_pos[1]))
            else:
                # If no direct candidate works, try a grid search in the free space?
                # For simplicity, if greedy fails, return None.
                # To improve success rate, we could try smaller steps or random perturbations here,
                # but for initialization, failing is okay as long as we have multiple starts.
                return None
        
        return s, cx, cy

    def local_search(s, cx, cy, max_iters=5000):
        """
        Improve the packing by moving/resizing one square at a time.
        """
        s = s.copy()
        cx = cx.copy()
        cy = cy.copy()
        
        current_sum = np.sum(s)
        best_sum = current_sum
        best_s = s.copy()
        best_cx = cx.copy()
        best_cy = cy.copy()
        
        no_improve_count = 0
        
        for _ in range(max_iters):
            improved = False
            # Try modifying each square
            for i in range(n):
                # Try resizing
                for delta_s in [0.01, 0.005, -0.01, -0.005, 0.02, -0.02]:
                    new_s = s[i] + delta_s
                    if new_s <= 0 or new_s > 1.0:
                        continue
                    
                    # Try to find a valid position for this new size
                    # We can try the current position first
                    # Then try shifting to resolve overlaps
                    
                    # Simple strategy: Keep center, check validity. If invalid, try to push away.
                    # A more robust local move: Randomly perturb position and size slightly.
                    pass # Skipping complex resize logic for speed, focusing on position moves first
                    
                # Try moving
                for dx in [-0.05, -0.02, 0, 0.02, 0.05]:
                    for dy in [-0.05, -0.02, 0, 0.02, 0.05]:
                        if dx == 0 and dy == 0:
                            continue
                        new_cx = cx[i] + dx
                        new_cy = cy[i] + dy
                        
                        # Check bounds
                        half = s[i] / 2
                        if new_cx - half < -tol or new_cx + half > 1 + tol:
                            continue
                        if new_cy - half < -tol or new_cy + half > 1 + tol:
                            continue
                        
                        # Check overlaps
                        valid = True
                        for j in range(n):
                            if i == j:
                                continue
                            t = (s[i] + s[j]) / 2 - tol
                            if abs(new_cx - cx[j]) < t and abs(new_cy - cy[j]) < t:
                                valid = False
                                break
                        
                        if valid:
                            # Calculate new sum (same since size didn't change, but we might resize in a more advanced version)
                            # Here we only move, so sum is same. 
                            # To improve sum, we need to allow resizing.
                            # Let's add a resize step if we can.
                            # Actually, just moving doesn't increase sum. 
                            # We need to try increasing size.
                            pass
                
                # Combined Move + Resize
                for delta_s in [0.005, 0.01, 0.02]:
                    new_s = s[i] + delta_s
                    if new_s > 1.0:
                        continue
                    half_new = new_s / 2
                    
                    # Try to find a valid spot for the larger square
                    # Start from current center
                    new_cx = cx[i]
                    new_cy = cy[i]
                    
                    # If current spot is invalid, try shifting
                    # Simple heuristic: try a few shifts
                    found_valid = False
                    for shift_x in [-0.1, -0.05, 0, 0.05, 0.1]:
                        for shift_y in [-0.1, -0.05, 0, 0.05, 0.1]:
                            tcx = cx[i] + shift_x
                            tcy = cy[i] + shift_y
                            
                            if tcx - half_new < -tol or tcx + half_new > 1 + tol:
                                continue
                            if tcy - half_new < -tol or tcy + half_new > 1 + tol:
                                continue
                            
                            valid = True
                            for j in range(n):
                                if i == j:
                                    continue
                                t = (new_s + s[j]) / 2 - tol
                                if abs(tcx - cx[j]) < t and abs(tcy - cy[j]) < t:
                                    valid = False
                                    break
                            
                            if valid:
                                # Accept
                                s[i] = new_s
                                cx[i] = tcx
                                cy[i] = tcy
                                improved = True
                                found_valid = True
                                break
                        if found_valid:
                            break
                    
                    if found_valid:
                        break # Move to next square
                
            if improved:
                current_sum = np.sum(s)
                if current_sum > best_sum + 1e-9:
                    best_sum = current_sum
                    best_s = s.copy()
                    best_cx = cx.copy()
                    best_cy = cy.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            if no_improve_count > 100:
                break
        
        return best_s, best_cx, best_cy

    # Main Logic: Multi-start
    best_global_sum = 0
    best_global_s = None
    best_global_cx = None
    best_global_cy = None
    
    np.random.seed(42)
    random.seed(42)
    
    # Define several size distributions to try
    # 1. Uniform-ish
    # 2. One big, rest small
    # 3. Two big, rest small
    # 4. Random
    
    size_configs = []
    
    # Config 1: 1 big (0.5), 4 med (0.25), 5 small (0.15)
    size_configs.append([0.5, 0.25, 0.25, 0.25, 0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    # Config 2: 2 big (0.4), 4 med (0.2), 4 small (0.15)
    size_configs.append([0.4, 0.4, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15, 0.15, 0.15])
    
    # Config 3: 5 med (0.3), 5 small (0.1)
    size_configs.append([0.3, 0.3, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    # Config 4: Randomized around 0.3
    base = 0.3
    for _ in range(10):
        size_configs.append([base + np.random.uniform(-0.1, 0.1) for _ in range(10)])
    
    # Config 5: Randomized around 0.25
    for _ in range(10):
        size_configs.append([0.25 + np.random.uniform(-0.1, 0.1) for _ in range(10)])
        
    # Limit total starts to manage time
    num_starts = 20
    
    for i in range(num_starts):
        if i < len(size_configs):
            sizes = size_configs[i]
        else:
            sizes = [np.random.uniform(0.1, 0.4) for _ in range(10)]
            
        # Shuffle the assignment of sizes to indices to create diversity in packing
        perm = np.random.permutation(n)
        shuffled_sizes = [sizes[perm[k]] for k in range(n)]
        
        pack_result = greedy_pack(shuffled_sizes)
        if pack_result is None:
            continue
            
        s, cx, cy = pack_result
        
        # Run Local Search
        ls_s, ls_cx, ls_cy = local_search(s, cx, cy, max_iters=2000)
        
        # Verify validity
        if is_valid(ls_s, ls_cx, ls_cy):
            total_sum = np.sum(ls_s)
            if total_sum > best_global_sum:
                best_global_sum = total_sum
                best_global_s = ls_s
                best_global_cx = ls_cx
                best_global_cy = ls_cy
                # print(f"New best: {total_sum}") # Debug

    # Fallback if all failed (should not happen)
    if best_global_s is None:
        # Use the simple grid from the original code as fallback
        s = np.array([0.3]*10)
        # Arrange in 3x3 grid + 1
        # This is a rough fallback
        cx = np.zeros(10)
        cy = np.zeros(10)
        # 3x3 grid of 0.3 doesn't fit in 1x1 (0.9x0.9 fits).
        # Let's just use the first successful greedy pack or a very safe small size.
        s = np.array([0.2]*10)
        cx = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.1, 0.3, 0.5, 0.7, 0.9])
        cy = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3, 0.3])
        # This might overlap. Let's use a 5x2 grid of 0.2.
        # Width 1.0, Height 0.4.
        cx = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.1, 0.3, 0.5, 0.7, 0.9])
        cy = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3, 0.3])
        best_global_s = s
        best_global_cx = cx
        best_global_cy = cy

    centers = np.stack([best_global_cx, best_global_cy], axis=1).astype(np.float64)
    sides = best_global_s.astype(np.float64)
    
    return centers, sides

# EVOLVE-BLOCK-END
