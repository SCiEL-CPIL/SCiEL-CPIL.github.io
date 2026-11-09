"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Sort boxes by volume descending, then by max dimension descending
    # This helps fill bins more efficiently
    indexed_boxes = [(i, boxes[i]) for i in range(n)]
    indexed_boxes.sort(key=lambda x: (-x[1][0]*x[1][1]*x[1][2], -max(x[1])))
    
    # We'll use a shelf-based approach but with multiple strategies and local search
    # Strategy 1: Best-fit decreasing with multiple shelf orientations
    # Strategy 2: Try different orderings
    
    best_placements = None
    best_score = float('inf')
    
    def compute_score(placements):
        """Compute the score: B + 0.1*(1 - last_bin_occupied_volume/container_volume)"""
        if not placements:
            return 0.0
        bin_volumes = {}
        for b, x, y, z, w, h, d in placements:
            if b not in bin_volumes:
                bin_volumes[b] = 0
            bin_volumes[b] += w * h * d
        B = len(bin_volumes)
        max_bin = max(bin_volumes.keys())
        last_vol = bin_volumes[max_bin]
        container_vol = W * H * D
        if container_vol == 0:
            return float(B)
        return B + 0.1 * (1 - last_vol / container_vol)
    
    def is_valid_placement(bin_placements, x, y, z, w, h, d):
        """Check if placing a box at (x,y,z) with dims (w,h,d) is valid in a bin"""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (bx, by, bz, bw, bh, bd) in bin_placements:
            # Check overlap: boxes overlap if they intersect on all 3 axes
            if not (x + w <= bx or bx + bw <= x):
                if not (y + h <= by or by + bh <= y):
                    if not (z + d <= bz or bz + bd <= z):
                        return False
        return True
    
    def get_candidate_positions(bin_placements, w, h, d):
        """Generate candidate positions for a box in a bin using extreme points"""
        candidates = [(0, 0, 0)]
        for (bx, by, bz, bw, bh, bd) in bin_placements:
            # Positions adjacent to existing boxes
            candidates.append((bx + bw, by, bz))
            candidates.append((bx, by + bh, bz))
            candidates.append((bx, by, bz + bd))
            # Diagonal positions
            candidates.append((bx + bw, by + bh, bz))
            candidates.append((bx + bw, by, bz + bd))
            candidates.append((bx, by + bh, bz + bd))
            candidates.append((bx + bw, by + bh, bz + bd))
        
        # Filter valid candidates
        valid = []
        seen = set()
        for (x, y, z) in candidates:
            if (x, y, z) in seen:
                continue
            seen.add((x, y, z))
            if is_valid_placement(bin_placements, x, y, z, w, h, d):
                valid.append((x, y, z))
        
        if not valid:
            # Fallback: try grid search
            for x in range(0, W - w + 1):
                for y in range(0, H - h + 1):
                    for z in range(0, D - d + 1):
                        if is_valid_placement(bin_placements, x, y, z, w, h, d):
                            valid.append((x, y, z))
                            break
                    if valid:
                        break
                if valid:
                    break
        
        return valid
    
    def solve_with_order(order, bins_list, bin_volumes, bin_placements, max_bins):
        """
        Try to place boxes in the given order into the bins.
        Returns (placements, success) where placements is list of (bin, x, y, z, w, h, d)
        """
        placements = []
        
        for idx in order:
            w, h, d = boxes[idx]
            placed = False
            
            # Try each bin
            for b in range(len(bins_list)):
                if len(bins_list) < max_bins or b < max_bins - 1:
                    # Try to place in this bin
                    candidates = get_candidate_positions(bin_placements[b], w, h, d)
                    # Sort candidates by (x, y, z) to prefer lower-left-front
                    candidates.sort()
                    for (x, y, z) in candidates:
                        if is_valid_placement(bin_placements[b], x, y, z, w, h, d):
                            # Place it
                            bin_placements[b].append((x, y, z, w, h, d))
                            bin_volumes[b] += w * h * d
                            placements.append((b, x, y, z, w, h, d))
                            placed = True
                            break
                    if placed:
                        break
                
                # If we're at the last allowed bin and couldn't place, try next bin
                if not placed and b == len(bins_list) - 1 and len(bins_list) < max_bins:
                    # Open a new bin
                    new_bin = len(bins_list)
                    bins_list.append(new_bin)
                    bin_placements.append([])
                    bin_volumes.append(0)
                    # Try to place in the new bin
                    candidates = get_candidate_positions(bin_placements[new_bin], w, h, d)
                    candidates.sort()
                    for (x, y, z) in candidates:
                        if is_valid_placement(bin_placements[new_bin], x, y, z, w, h, d):
                            bin_placements[new_bin].append((x, y, z, w, h, d))
                            bin_volumes[new_bin] += w * h * d
                            placements.append((new_bin, x, y, z, w, h, d))
                            placed = True
                            break
                    if placed:
                        break
            
            if not placed:
                # Couldn't place this box - return failure
                return None, False
        
        return placements, True
    
    # Binary search on number of bins
    total_vol = sum(w*h*d for w,h,d in boxes)
    container_vol = W * H * D
    min_bins_vol = math.ceil(total_vol / container_vol) if container_vol > 0 else n
    
    # Try increasing number of bins from lower bound
    for max_bins in range(min_bins_vol, n + 1):
        if time.time() - start_time > deadline - 5:
            break
        
        # Try multiple orderings
        orderings = [
            [i for i, _ in indexed_boxes],  # Volume descending
            list(range(n)),  # Original order
        ]
        
        # Add some random-ish orderings based on dimensions
        orderings.append(sorted(range(n), key=lambda i: (-boxes[i][0], -boxes[i][1], -boxes[i][2])))
        orderings.append(sorted(range(n), key=lambda i: (boxes[i][0], boxes[i][1], boxes[i][2])))
        
        found = False
        for order in orderings:
            if time.time() - start_time > deadline - 5:
                break
            
            bins_list = []
            bin_volumes = []
            bin_placements = []
            
            placements, success = solve_with_order(order, bins_list, bin_volumes, bin_placements, max_bins)
            
            if success:
                score = compute_score(placements)
                if score < best_score:
                    best_score = score
                    best_placements = placements[:]
                found = True
                break
        
        if found:
            break
    
    # If no valid placement found, fall back to simple shelf heuristic
    if best_placements is None:
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
        best_placements = placements
    
    return best_placements
# EVOLVE-BLOCK-END
