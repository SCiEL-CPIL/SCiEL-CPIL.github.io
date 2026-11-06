"""Improved 3D Bin Packing with Multi-Start Heuristics and MSP."""
import time
import sys

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Helper to calculate volume
    def vol(b):
        return b[0]*b[1]*b[2]
    
    # Define different sorting strategies
    # 1. Volume Descending
    # 2. Surface Area Descending (proxy for "largeness")
    # 3. Max Dimension Descending
    # 4. Volume Descending, tie-break by Max Dim
    
    def get_sort_key(strategy):
        if strategy == 0:
            return lambda x: -vol(x[1])
        elif strategy == 1:
            return lambda x: -(2*(x[1][0]*x[1][1] + x[1][1]*x[1][2] + x[1][0]*x[1][2]))
        elif strategy == 2:
            return lambda x: -max(x[1])
        else:
            return lambda x: (-vol(x[1]), -max(x[1]))
            
    def new_bin():
        return [(0, 0, 0, W, H, D)]
    
    def split_space(space, box_pos, box_dims):
        sx, sy, sz, sw, sh, sd = space
        bx, by, bz = box_pos
        bw, bh, bd = box_dims
        
        new_spaces = []
        
        # Region below the box in z
        if bz > sz:
            new_spaces.append((sx, sy, sz, sw, sh, bz - sz))
        
        # Region above the box in z
        if bz + bd < sz + sd:
            new_spaces.append((sx, sy, bz + bd, sw, sh, sz + sd - bz - bd))
        
        z_slab_start = max(bz, sz)
        z_slab_end = min(bz + bd, sz + sd)
        if z_slab_start < z_slab_end:
            z_slab_depth = z_slab_end - z_slab_start
            
            if by > sy:
                new_spaces.append((sx, sy, z_slab_start, sw, by - sy, z_slab_depth))
            
            if by + bh < sy + sh:
                new_spaces.append((sx, by + bh, z_slab_start, sw, sy + sh - by - bh, z_slab_depth))
            
            y_slab_start = max(by, sy)
            y_slab_end = min(by + bh, sy + sh)
            if y_slab_start < y_slab_end:
                y_slab_height = y_slab_end - y_slab_start
                
                if bx > sx:
                    new_spaces.append((sx, y_slab_start, z_slab_start, bx - sx, y_slab_height, z_slab_depth))
                
                if bx + bw < sx + sw:
                    new_spaces.append((bx + bw, y_slab_start, z_slab_start, sx + sw - bx - bw, y_slab_height, z_slab_depth))
        
        return new_spaces
    
    def can_place_in_space(space, w, h, d):
        return w <= space[3] and h <= space[4] and d <= space[5]
    
    def find_best_position(bin_free_spaces, w, h, d):
        """
        Find best position. Score: Minimize wasted volume in the specific space, 
        then minimize x+y+z (BLF).
        """
        best_score = float('inf')
        best_pos = None
        best_space_idx = None
        
        for idx, space in enumerate(bin_free_spaces):
            if not can_place_in_space(space, w, h, d):
                continue
            
            sx, sy, sz, sw, sh, sd = space
            bx, by, bz = sx, sy, sz
            
            # Calculate the volume of the new free spaces if we place here
            # This is an approximation of "waste" in this specific slot
            # Waste = (sw-w)*sh*sd + (sh-h)*sw*sd + (sd-d)*sw*sh is not quite right for the split parts
            # A better metric for "tightness" is the sum of the dimensions of the resulting free spaces
            # Or simply: prefer smaller spaces.
            
            # Let's use a score that favors tighter fits (less residual space in the box's local area)
            # Residual space in x, y, z directions
            # If we place at (sx, sy, sz), the remaining space in this cuboid is roughly:
            # (sw-w) in x, (sh-h) in y, (sd-d) in z.
            # A common heuristic is to minimize the sum of the "gaps".
            gap_score = (sw - w) + (sh - h) + (sd - d)
            
            # BLF score
            pos_score = bx + by + bz
            
            # Combine: Prioritize tight fit (lower gap_score), then BLF
            # We scale gap_score to be dominant
            score = gap_score * 100.0 + pos_score
            
            if score < best_score:
                best_score = score
                best_pos = (bx, by, bz)
                best_space_idx = idx
        
        return best_pos, best_space_idx
    
    def run_packing(sorted_indices):
        """Run the packing algorithm with a specific box order. Returns placements."""
        bins = []
        placements = [None] * n
        
        for orig_idx in sorted_indices:
            w, h, d = boxes[orig_idx]
            
            placed = False
            
            # Try existing bins
            # Sort bins by "fullness" (least free volume first) to try tightest bins first?
            # Or just iterate. Iterating in order of creation is standard FFD.
            # Let's try bins in reverse order of creation (most recently opened might have space? 
            # No, usually first fit is better for bin packing to keep early bins full).
            # We will iterate bins 0 to len(bins)-1.
            
            for b_idx in range(len(bins)):
                free_spaces = bins[b_idx]
                pos, space_idx = find_best_position(free_spaces, w, h, d)
                if pos is not None:
                    placements[orig_idx] = (b_idx, pos[0], pos[1], pos[2], w, h, d)
                    
                    old_space = free_spaces[space_idx]
                    new_spaces = split_space(old_space, pos, (w, h, d))
                    free_spaces[space_idx:space_idx+1] = new_spaces
                    placed = True
                    break
            
            if not placed:
                b_idx = len(bins)
                bins.append(new_bin())
                placements[orig_idx] = (b_idx, 0, 0, 0, w, h, d)
                new_spaces = split_space((0, 0, 0, W, H, D), (0, 0, 0), (w, h, d))
                bins[b_idx] = new_spaces
        
        return placements, bins
    
    def calculate_score(placements, bins):
        """Calculate the score for a set of placements."""
        if not placements or any(p is None for p in placements):
            return float('inf')
        
        occupied_bins = set(p[0] for p in placements)
        B = len(occupied_bins)
        
        if B == 0:
            return 0.0
            
        last_bin = max(occupied_bins)
        
        # Calculate occupied volume in last bin
        last_bin_vol = 0
        for p in placements:
            if p[0] == last_bin:
                last_bin_vol += p[4] * p[5] * p[6]
                
        container_vol = W * H * D
        fraction = last_bin_vol / container_vol if container_vol > 0 else 0.0
        
        score = B + 0.1 * (1 - fraction)
        return score

    # Multi-start search
    best_placements = None
    best_score = float('inf')
    
    # Define strategies
    strategies = [
        "volume", 
        "surface", 
        "max_dim", 
        "vol_maxdim"
    ]
    
    # We will try each strategy. If time runs out, we return the best found so far.
    # To be safe, we limit the number of strategies we try based on remaining time.
    # Assuming ~5-10ms per box per strategy, 40 boxes is ~400ms per strategy.
    # 4 strategies = ~1.6s. We have 115s. We can even do more or add random restarts.
    
    for strat_name in strategies:
        if time.time() > deadline - 1.0:
            break
            
        if strat_name == "volume":
            key_func = lambda x: -vol(x[1])
        elif strat_name == "surface":
            key_func = lambda x: -(2*(x[1][0]*x[1][1] + x[1][1]*x[1][2] + x[1][0]*x[1][2]))
        elif strat_name == "max_dim":
            key_func = lambda x: -max(x[1])
        else:
            key_func = lambda x: (-vol(x[1]), -max(x[1]))
            
        # Sort indices
        indexed_boxes = list(enumerate(boxes))
        indexed_boxes.sort(key=key_func)
        sorted_indices = [idx for idx, _ in indexed_boxes]
        
        # Run packing
        placements, bins = run_packing(sorted_indices)
        
        # Calculate score
        score = calculate_score(placements, bins)
        
        if score < best_score:
            best_score = score
            best_placements = placements
            
        # If we found a perfect score (1 bin, full), we can stop?
        # Score 1.0 is 1 bin + 0.1*(1-1) = 1.0.
        # If B=1 and fraction=1.0, score is 1.0.
        # If B=1 and fraction < 1.0, score > 1.0.
        # We can't really stop unless score is theoretically minimal?
        # Minimal B is 1. Minimal score for B=1 is 1.0.
        # So if best_score == 1.0, we can stop.
        if best_score <= 1.0 + 1e-9:
            break

    # If no strategy ran (should not happen), fallback to first one
    if best_placements is None:
        indexed_boxes = list(enumerate(boxes))
        indexed_boxes.sort(key=lambda x: -vol(x[1]))
        sorted_indices = [idx for idx, _ in indexed_boxes]
        best_placements, _ = run_packing(sorted_indices)
        
    # Fallback check
    for i in range(n):
        if best_placements[i] is None:
            w, h, d = boxes[i]
            best_placements[i] = (0, 0, 0, 0, w, h, d) # This is invalid if it overlaps, but better than None
            
    return best_placements
