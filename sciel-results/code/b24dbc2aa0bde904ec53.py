"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    # Compute total volume and container volume
    total_vol = sum(w*h*d for w,h,d in boxes)
    cont_vol = W*H*D
    
    # Try multiple strategies and keep the best
    best_placement = None
    best_score = float('inf')
    
    # Strategy 1: Sort by volume descending, place with best-fit decreasing
    # Strategy 2: Sort by max dimension descending
    # Strategy 3: Sort by height descending (for shelf-like packing)
    # Strategy 4: Random orderings
    
    def compute_score(placement):
        """Compute the score: mean B + 0.1*(1 - last-bin occupied volume fraction)"""
        if not placement:
            return 0.0
        bins_used = {}
        for (b, x, y, z, w, h, d) in placement:
            if b not in bins_used:
                bins_used[b] = 0
            bins_used[b] += w*h*d
        B = len(bins_used)
        if B == 0:
            return 0.0
        last_bin = max(bins_used.keys())
        last_vol = bins_used[last_bin]
        return B + 0.1 * (1 - last_vol / cont_vol)
    
    def try_placement(ordered_boxes, placement_fn):
        """Try to place boxes in given order using placement_fn, return placement or None if failed"""
        placement = []
        bins = []  # list of lists of (x,y,z,w,h,d)
        
        for idx, (w, h, d) in enumerate(ordered_boxes):
            placed = False
            # Try existing bins first
            for b_idx, bin_boxes in enumerate(bins):
                pos = placement_fn(bin_boxes, w, h, d, W, H, D)
                if pos is not None:
                    x, y, z = pos
                    bins[b_idx].append((x, y, z, w, h, d))
                    placement.append((b_idx, x, y, z, w, h, d))
                    placed = True
                    break
            
            if not placed:
                # Try new bin
                pos = placement_fn([], w, h, d, W, H, D)
                if pos is not None:
                    x, y, z = pos
                    bins.append([(x, y, z, w, h, d)])
                    placement.append((len(bins)-1, x, y, z, w, h, d))
                    placed = True
                else:
                    return None  # Can't place this box
            
        return placement
    
    def check_overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        """Check if two boxes overlap"""
        if x1 + w1 <= x2 or x2 + w2 <= x1:
            return False
        if y1 + h1 <= y2 or y2 + h2 <= y1:
            return False
        if z1 + d1 <= z2 or z2 + d2 <= z1:
            return False
        return True
    
    def find_position_brute(bin_boxes, w, h, d, W, H, D, max_candidates=200):
        """Find a position in the bin using candidate positions from existing box faces"""
        # Candidate positions are derived from existing box edges
        candidates = set()
        candidates.add((0, 0, 0))
        
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            # Positions adjacent to existing boxes
            candidates.add((bx + bw, by, bz))
            candidates.add((bx, by + bh, bz))
            candidates.add((bx, by, bz + bd))
            # Also try combinations
            candidates.add((bx + bw, by + bh, bz))
            candidates.add((bx + bw, by, bz + bd))
            candidates.add((bx, by + bh, bz + bd))
            candidates.add((bx + bw, by + bh, bz + bd))
        
        # Limit candidates for speed
        if len(candidates) > max_candidates:
            # Keep only those within bounds
            valid = [(x, y, z) for (x, y, z) in candidates if x + w <= W and y + h <= H and z + d <= D]
            if len(valid) > max_candidates:
                candidates = set(valid[:max_candidates])
        
        best_pos = None
        best_score = float('inf')
        
        for (x, y, z) in candidates:
            if x + w > W or y + h > H or z + d > D:
                continue
            # Check overlap with all boxes in bin
            overlap = False
            for (bx, by, bz, bw, bh, bd) in bin_boxes:
                if check_overlap(x, y, z, w, h, d, bx, by, bz, bw, bh, bd):
                    overlap = True
                    break
            if not overlap:
                # Score: prefer lower z, then lower y, then lower x
                pos_score = z * W * H + y * W + x
                if pos_score < best_score:
                    best_score = pos_score
                    best_pos = (x, y, z)
        
        return best_pos
    
    def find_position_shelf(bin_boxes, w, h, d, W, H, D):
        """Shelf-based placement: find a shelf where the box fits"""
        if not bin_boxes:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None
        
        # Build shelf structure
        # Shelves are horizontal planes at various z levels
        shelves = {}  # z -> list of (x, y, w, h)
        
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            z_key = bz
            if z_key not in shelves:
                shelves[z_key] = []
            shelves[z_key].append((bx, by, bw, bh))
        
        # Try to place on existing shelves or create new shelf
        best_pos = None
        best_score = float('inf')
        
        for z in sorted(shelves.keys()):
            shelf_boxes = shelves[z]
            # Find gaps in this shelf
            # Simple approach: try placing at x=0, then after each box
            tried_positions = set()
            for bx, by, bw, bh in shelf_boxes:
                # Try placing to the right of this box at same y
                new_x = bx + bw
                new_y = by
                if new_x + w <= W and by + h <= H and z + d <= D:
                    key = (new_x, new_y, z)
                    if key not in tried_positions:
                        tried_positions.add(key)
                        # Check overlap
                        overlap = False
                        for (bx2, by2, bz2, bw2, bh2, bd2) in bin_boxes:
                            if check_overlap(new_x, new_y, z, w, h, d, bx2, by2, bz2, bw2, bh2, bd2):
                                overlap = True
                                break
                        if not overlap:
                            pos_score = z * W * H + new_y * W + new_x
                            if pos_score < best_score:
                                best_score = pos_score
                                best_pos = (new_x, new_y, z)
                # Try placing above this box
                new_x = bx
                new_y = by
                new_z = bz + bd
                if new_x + w <= W and new_y + h <= H and new_z + d <= D:
                    key = (new_x, new_y, new_z)
                    if key not in tried_positions:
                        tried_positions.add(key)
                        overlap = False
                        for (bx2, by2, bz2, bw2, bh2, bd2) in bin_boxes:
                            if check_overlap(new_x, new_y, new_z, w, h, d, bx2, by2, bz2, bw2, bh2, bd2):
                                overlap = True
                                break
                        if not overlap:
                            pos_score = new_z * W * H + new_y * W + new_x
                            if pos_score < best_score:
                                best_score = pos_score
                                best_pos = (new_x, new_y, new_z)
            
            # Also try at origin of this shelf
            if (0, 0, z) not in tried_positions:
                if w <= W and h <= H and z + d <= D:
                    overlap = False
                    for (bx2, by2, bz2, bw2, bh2, bd2) in bin_boxes:
                        if check_overlap(0, 0, z, w, h, d, bx2, by2, bz2, bw2, bh2, bd2):
                            overlap = True
                            break
                    if not overlap:
                        pos_score = z * W * H
                        if pos_score < best_score:
                            best_score = pos_score
                            best_pos = (0, 0, z)
        
        # Try new shelf at z=0 if not tried
        if best_pos is None or best_pos[2] != 0:
            if w <= W and h <= H and d <= D:
                overlap = False
                for (bx, by, bz, bw, bh, bd) in bin_boxes:
                    if check_overlap(0, 0, 0, w, h, d, bx, by, bz, bw, bh, bd):
                        overlap = True
                        break
                if not overlap:
                    return (0, 0, 0)
        
        return best_pos
    
    # Strategy functions
    def strategy_volume_desc():
        return sorted(boxes, key=lambda b: b[0]*b[1]*b[2], reverse=True)
    
    def strategy_max_dim_desc():
        return sorted(boxes, key=lambda b: max(b), reverse=True)
    
    def strategy_height_desc():
        return sorted(boxes, key=lambda b: b[1], reverse=True)
    
    def strategy_width_desc():
        return sorted(boxes, key=lambda b: b[0], reverse=True)
    
    def strategy_depth_desc():
        return sorted(boxes, key=lambda b: b[2], reverse=True)
    
    def strategy_random():
        ordered = list(boxes)
        random.shuffle(ordered)
        return ordered
    
    # Try deterministic strategies first
    strategies = [
        (strategy_volume_desc, find_position_brute),
        (strategy_max_dim_desc, find_position_brute),
        (strategy_height_desc, find_position_shelf),
        (strategy_width_desc, find_position_brute),
        (strategy_depth_desc, find_position_brute),
    ]
    
    for order_fn, place_fn in strategies:
        if time.time() > deadline:
            break
        ordered = order_fn()
        placement = try_placement(ordered, place_fn)
        if placement is not None:
            score = compute_score(placement)
            if score < best_score:
                best_score = score
                best_placement = placement
    
    # If we have time, try random strategies
    if time.time() < deadline - 5.0 and best_placement is not None:
        n_random = min(20, int((deadline - time.time()) / 1.0))
        for _ in range(n_random):
            if time.time() > deadline:
                break
            ordered = strategy_random()
            placement = try_placement(ordered, find_position_brute)
            if placement is not None:
                score = compute_score(placement)
                if score < best_score:
                    best_score = score
                    best_placement = placement
    
    if best_placement is None:
        # Fallback: simple shelf packing
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements
    
    return best_placement
# EVOLVE-BLOCK-END
