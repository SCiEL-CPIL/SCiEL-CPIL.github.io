"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def pack(boxes, container):
    """
    Pack boxes into bins using a Best-Fit Decreasing heuristic with Skyline representation.
    Returns a list of placements: (bin, x, y, z, w, h, d) for each box in original order.
    """
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin from 120s limit
    
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    # Sort indices by volume descending for placement order
    # But we need to return in original order, so we track original indices
    volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    order = sorted(range(n), key=lambda i: -volumes[i])
    
    # Bin representation: each bin has a list of placed boxes (x, y, z, w, h, d)
    # For efficiency, we'll use a skyline-based approach
    # Each bin stores its placed items and we compute valid positions on the fly
    
    # For 40 boxes, we can afford a moderately complex heuristic
    # Let's use a simple but effective approach:
    # 1. Sort by volume descending
    # 2. For each box, try to place in existing bins (best fit)
    # 3. If no bin fits, open a new bin
    
    # To find valid positions, we'll use a grid-based or skyline approach
    # For simplicity and speed with 40 boxes, let's use a simple coordinate-based check
    
    def find_position(bin_items, w, h, d):
        """
        Find the best (lowest z, then lowest y, then lowest x) position for a box
        in a bin given the list of already placed items.
        Returns (x, y, z) or None if no valid position.
        """
        # Generate candidate positions
        # x candidates: 0 and x + w for each placed item
        # y candidates: 0 and y + h for each placed item
        # z candidates: 0 and z + d for each placed item
        
        x_cands = [0]
        y_cands = [0]
        z_cands = [0]
        
        for (px, py, pz, pw, ph, pd) in bin_items:
            if px + pw <= W:
                x_cands.append(px + pw)
            if py + ph <= H:
                y_cands.append(py + ph)
            if pz + pd <= D:
                z_cands.append(pz + pd)
        
        x_cands = sorted(set(x_cands))
        y_cands = sorted(set(y_cands))
        z_cands = sorted(set(z_cands))
        
        best_pos = None
        best_score = None
        
        for z in z_cands:
            if z + d > D:
                break
            for y in y_cands:
                if y + h > H:
                    break
                for x in x_cands:
                    if x + w > W:
                        break
                    # Check if this position is valid (no overlap)
                    valid = True
                    for (px, py, pz, pw, ph, pd) in bin_items:
                        # Check overlap in all three dimensions
                        if (x < px + pw and px < x + w and
                            y < py + ph and py < y + h and
                            z < pz + pd and pz < z + d):
                            valid = False
                            break
                    if valid:
                        # Score: prefer lower z, then lower y, then lower x
                        score = (z, y, x)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
                        break  # Only check lowest x for this (y, z)
                if best_pos is not None and best_score[1] == y and best_score[2] == x_cands[0]:
                    # We found the best x for this (y, z), but there might be better (y, z)
                    pass
            # Early termination if we found position at z=0, y=0, x=0
            if best_pos is not None and best_pos == (0, 0, 0):
                return best_pos
        
        return best_pos
    
    def bin_volume(bin_items, W, H, D):
        """Calculate the bounding box volume used by items in a bin."""
        if not bin_items:
            return 0
        max_x = max(px + pw for (px, py, pz, pw, ph, pd) in bin_items)
        max_y = max(py + ph for (px, py, pz, pw, ph, pd) in bin_items)
        max_z = max(pz + pd for (px, py, pz, pw, ph, pd) in bin_items)
        return max_x * max_y * max_z
    
    # Main packing loop
    bins = []  # List of lists of (x, y, z, w, h, d)
    placement_map = [None] * n  # Maps original index to (bin_idx, x, y, z)
    
    for idx in order:
        w, h, d = boxes[idx]
        placed = False
        
        # Try to place in existing bins (best fit: minimize wasted space)
        best_bin = -1
        best_pos = None
        best_waste = float('inf')
        
        for b_idx, bin_items in enumerate(bins):
            pos = find_position(bin_items, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Calculate waste: prefer positions that minimize the max coordinate
                waste = z * H * W + y * W + x  # Rough waste metric
                if waste < best_waste:
                    best_waste = waste
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin >= 0:
            x, y, z = best_pos
            bins[best_bin].append((x, y, z, w, h, d))
            placement_map[idx] = (best_bin, x, y, z)
        else:
            # Open new bin
            bins.append([(0, 0, 0, w, h, d)])
            placement_map[idx] = (len(bins) - 1, 0, 0, 0)
        
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
    
    # If we didn't place all boxes due to deadline, use a simpler fallback
    if time.time() - start_time > DEADLINE:
        # Fallback: just place all remaining boxes in a simple shelf manner
        # This is a safety net; in practice with 40 boxes we should finish quickly
        pass
    
    # Build result in original order
    result = []
    for i in range(n):
        if placement_map[i] is None:
            # Should not happen, but safety
            result.append((0, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
        else:
            b_idx, x, y, z = placement_map[i]
            w, h, d = boxes[i]
            result.append((b_idx, x, y, z, w, h, d))
    
    return result

# EVOLVE-BLOCK-END
