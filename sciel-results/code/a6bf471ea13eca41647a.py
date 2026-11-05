import time
import random
import math

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # 1. Initialization: Fast Shelf-Based Heuristic
    # This provides a valid starting point quickly.
    # Sort by volume descending for better initial packing
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
    
    # State: list of bins. Each bin is a list of (box_idx, x, y, z, w, h, d)
    bins = []
    
    def can_place_in_bin(bin_idx, x, y, z, w, h, d, box_idx):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (b_idx, ox, oy, oz, ow, oh, od) in bins[bin_idx]:
            if b_idx == box_idx:
                continue
            # Check overlap
            if not (x + w <= ox or ox + ow <= x or
                    y + h <= oy or oy + oh <= y or
                    z + d <= oz or oz + od <= z):
                return False
        return True

    def find_shelf_position(bin_idx, w, h, d, box_idx):
        # Simple shelf logic: find the lowest z where it fits, then lowest y, then lowest x
        # To speed up, we can just check a few key positions or use a simple scan.
        # For initialization speed, we'll try (0,0,z) for increasing z, then (x,0,z), etc.
        # Actually, a simple "first fit" in a grid of potential spots derived from existing boxes is better.
        # But for init, let's keep it very fast.
        
        # Try to place at (0,0,z) for z in [0, max_z]
        # Then (x, 0, z) ...
        
        # A faster approximation for init:
        # 1. Get max heights per (x,y) column? Too complex for init.
        # 2. Just try a set of candidate positions.
        
        candidates = [(0,0,0)]
        max_z = 0
        for (b_idx, ox, oy, oz, ow, oh, od) in bins[bin_idx]:
            if b_idx == box_idx: continue
            if oz + od > max_z: max_z = oz + od
            candidates.append((ox + ow, oy, oz))
            candidates.append((ox, oy + oh, oz))
            candidates.append((ox, oy, oz + od))
            # Limit candidates to keep init fast
            if len(candidates) > 50: break
            
        # Add some random-ish or grid-based positions if bin is not empty
        if bins[bin_idx]:
            # Try bottom of the bin at various x,y
            for i in range(0, W, max(1, w//2)):
                for j in range(0, H, max(1, h//2)):
                    if i + w <= W and j + h <= H:
                        candidates.append((i, j, 0))
            if len(candidates) > 200: break

        # Deduplicate and sort by (z, y, x) to prefer bottom-left
        unique_candidates = list(set(candidates))
        unique_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        
        for cx, cy, cz in unique_candidates:
            if can_place_in_bin(bin_idx, cx, cy, cz, w, h, d, box_idx):
                return cx, cy, cz
        return None

    def place_box_init(box_idx, w, h, d):
        best_bin = -1
        best_pos = None
        best_score = float('inf')
        
        for i in range(len(bins)):
            pos = find_shelf_position(i, w, h, d, box_idx)
            if pos is not None:
                x, y, z = pos
                score = z * 1000 + y * 10 + x # Prefer low z, then y, then x
                if score < best_score:
                    best_score = score
                    best_bin = i
                    best_pos = pos
        
        if best_bin != -1:
            bins[best_bin].append((box_idx, best_pos[0], best_pos[1], best_pos[2], w, h, d))
        else:
            new_bin_idx = len(bins)
            bins.append([(box_idx, 0, 0, 0, w, h, d)])
            best_bin = new_bin_idx
            best_pos = (0,0,0)
            
        return best_bin, best_pos

    # Run initialization
    placements = [None] * n
    for idx, (w, h, d) in indexed_boxes:
        b, pos = place_box_init(idx, w, h, d)
        placements[idx] = (b, pos[0], pos[1], pos[2], w, h, d)
        
    # 2. Local Search / Simulated Annealing
    # Objective: Minimize B + 0.1 * (1 - last_bin_fill)
    # B is number of occupied bins.
    # last_bin_fill is volume of last bin / container volume.
    
    def get_score():
        if not bins:
            return float('inf')
        bin_vols = [0] * len(bins)
        for i in range(len(bins)):
            vol = 0
            for (b_idx, x, y, z, w, h, d) in bins[i]:
                vol += w * h * d
            bin_vols[i] = vol
        
        # Find last bin (highest index with content)
        last_bin_idx = -1
        for i in range(len(bins) - 1, -1, -1):
            if bin_vols[i] > 0:
                last_bin_idx = i
                break
                
        B = last_bin_idx + 1
        if last_bin_idx == -1:
            return 0.0 # No boxes?
            
        last_vol = bin_vols[last_bin_idx]
        cont_vol = W * H * D
        fill = last_vol / cont_vol if cont_vol > 0 else 0
        return B + 0.1 * (1 - fill)

    current_score = get_score()
    best_score = current_score
    best_placements = [p[:] for p in placements]
    
    # Precompute box volumes for quick access
    box_vols = [w*h*d for w,h,d in boxes]
    
    # Helper to get box dimensions from index
    def get_dims(idx):
        return boxes[idx]

    # Move types:
    # 1. Move box i from bin a to bin b (if valid)
    # 2. Move box i to a new random valid position in same bin
    
    def try_move_box(box_idx, target_bin_idx, x, y, z):
        """
        Attempts to move box_idx to (target_bin_idx, x, y, z).
        Returns True if successful, False otherwise.
        Assumes box_idx is currently in some bin.
        """
        # Find current bin
        current_bin_idx = -1
        current_pos = None
        for i in range(len(bins)):
            for j, (b_idx, ox, oy, oz, ow, oh, od) in enumerate(bins[i]):
                if b_idx == box_idx:
                    current_bin_idx = i
                    current_pos = (ox, oy, oz)
                    break
            if current_bin_idx != -1:
                break
                
        if current_bin_idx == -1:
            return False
            
        w, h, d = get_dims(box_idx)
        
        # Check if new position is valid in target bin
        if not can_place_in_bin(target_bin_idx, x, y, z, w, h, d, box_idx):
            return False
            
        # Perform move
        # Remove from current
        bins[current_bin_idx] = [(b_idx, ox, oy, oz, ow, oh, od) for (b_idx, ox, oy, oz, ow, oh, od) in bins[current_bin_idx] if b_idx != box_idx]
        
        # Add to target
        bins[target_bin_idx].append((box_idx, x, y, z, w, h, d))
        
        # Update placements
        placements[box_idx] = (target_bin_idx, x, y, z, w, h, d)
        
        return True

    def revert_move(box_idx, original_bin_idx, original_pos):
        """Reverts the last move for box_idx."""
        w, h, d = get_dims(box_idx)
        # Find where it is now
        current_bin_idx = -1
        current_pos = None
        for i in range(len(bins)):
            for j, (b_idx, ox, oy, oz, ow, oh, od) in enumerate(bins[i]):
                if b_idx == box_idx:
                    current_bin_idx = i
                    current_pos = (ox, oy, oz)
                    break
            if current_bin_idx != -1:
                break
                
        if current_bin_idx == -1:
            return False
            
        # Remove from current
        bins[current_bin_idx] = [(b_idx, ox, oy, oz, ow, oh, od) for (b_idx, ox, oy, oz, ow, oh, od) in bins[current_bin_idx] if b_idx != box_idx]
        
        # Add to original
        bins[original_bin_idx].append((box_idx, original_pos[0], original_pos[1], original_pos[2], w, h, d))
        
        # Update placements
        placements[box_idx] = (original_bin_idx, original_pos[0], original_pos[1], original_pos[2], w, h, d)
        
        return True

    # Simulated Annealing Loop
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    iterations = 0
    
    while time.time() < deadline:
        iterations += 1
        if T < T_min:
            break
            
        # Pick a random box
        box_idx = random.randint(0, n - 1)
        w, h, d = get_dims(box_idx)
        
        # Find current bin and position
        curr_bin = -1
        curr_pos = None
        for i in range(len(bins)):
            for j, (b_idx, ox, oy, oz, ow, oh, od) in enumerate(bins[i]):
                if b_idx == box_idx:
                    curr_bin = i
                    curr_pos = (ox, oy, oz)
                    break
            if curr_bin != -1:
                break
                
        if curr_bin == -1:
            continue
            
        # Generate a neighbor move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to move to a different bin
            if len(bins) > 1:
                target_bin = random.randint(0, len(bins) - 1)
                if target_bin == curr_bin:
                    continue
                    
                # Find a valid position in target bin
                # Use a simplified search: try current position, then some random valid spots
                pos = find_shelf_position(target_bin, w, h, d, box_idx)
                if pos is None:
                    continue
                    
                # Perform move
                success = try_move_box(box_idx, target_bin, pos[0], pos[1], pos[2])
                if not success:
                    continue
                    
                new_score = get_score()
                delta = new_score - current_score
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_placements = [p[:] for p in placements]
                else:
                    # Revert
                    revert_move(box_idx, curr_bin, curr_pos)
                    
        else:
            # Try to move to a different position in the same bin
            # Generate a new position
            pos = find_shelf_position(curr_bin, w, h, d, box_idx)
            if pos is None:
                continue
                
            # If pos is same as curr_pos, skip
            if pos == curr_pos:
                continue
                
            success = try_move_box(box_idx, curr_bin, pos[0], pos[1], pos[2])
            if not success:
                continue
                
            new_score = get_score()
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_placements = [p[:] for p in placements]
            else:
                # Revert
                revert_move(box_idx, curr_bin, curr_pos)
                
        T *= alpha
        
    # Return the best placement found
    # Ensure we return a valid list of tuples
    final_placements = []
    for p in best_placements:
        if p is not None:
            final_placements.append(tuple(p))
            
    # If something went wrong and we have unplaced boxes, place them in new bins
    # This is a safety net
    placed_indices = set(p[0] for p in final_placements)
    for i in range(n):
        if i not in placed_indices:
            w, h, d = boxes[i]
            new_bin_idx = len(bins) if bins else 0
            # Just assign to a new bin to ensure validity, though score will be bad
            # But since we track best_placements from a valid state, this shouldn't happen
            # unless initialization failed completely.
            final_placements.append((new_bin_idx, 0, 0, 0, w, h, d))
            
    # Re-order to match original box indices?
    # The task says: "one placement ... per box, preserving its original ordered dimensions"
    # The return is a list of placements. The evaluator likely expects the i-th item to correspond to the i-th box in the input `boxes` list.
    # My `placements` array is indexed by original box index.
    # So I should return `best_placements` directly, ensuring it's a list of tuples.
    
    # Check if best_placements has correct length
    if len(best_placements) != n:
         # Fallback to current placements if best is corrupted
         best_placements = placements
         
    return [tuple(p) for p in best_placements]
