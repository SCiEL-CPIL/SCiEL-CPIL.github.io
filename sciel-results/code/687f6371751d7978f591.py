"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def pack(boxes, container):
    """
    3D bin packing using Best-Fit Decreasing with extreme-point placement.
    Tries multiple sorting strategies and returns the best result.
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    def solve_with_strategy(indices):
        """Solve packing for a given box ordering. Returns (num_bins, placements_map, last_bin_fill)."""
        bins = []  # list of lists of (x, y, z, w, h, d)
        placement_map = [None] * n
        bin_volumes = []  # volume used in each bin
        
        for idx in indices:
            w, h, d = boxes[idx]
            placed = False
            
            # Try existing bins first (best fit)
            best_bin = -1
            best_pos = None
            best_score = float('inf')
            
            for bi in range(len(bins)):
                # Generate candidate positions from extreme points
                x_set = {0}
                y_set = {0}
                z_set = {0}
                
                for (bx, by, bz, bw, bh, bd) in bins[bi]:
                    x_set.add(bx + bw)
                    y_set.add(by + bh)
                    z_set.add(bz + bd)
                
                # Filter to valid range
                x_cands = [x for x in x_set if 0 <= x and x + w <= W]
                y_cands = [y for y in y_set if 0 <= y and y + h <= H]
                z_cands = [z for z in z_set if 0 <= z and z + d <= D]
                
                found_in_bin = False
                for x in x_cands:
                    for y in y_cands:
                        for z in z_cands:
                            # Check overlap
                            overlap = False
                            for (bx, by, bz, bw, bh, bd) in bins[bi]:
                                if not (x + w <= bx or bx + bw <= x or
                                        y + h <= by or by + bh <= y or
                                        z + d <= bz or bz + bd <= z):
                                    overlap = True
                                    break
                            if not overlap:
                                # Valid position found
                                # Score: prefer positions that minimize wasted space
                                # Lower x, y, z is better (packs toward origin)
                                pos_score = x * (H * D) + y * D + z
                                if pos_score < best_score:
                                    best_score = pos_score
                                    best_bin = bi
                                    best_pos = (x, y, z)
                                found_in_bin = True
                                break
                        if found_in_bin:
                            break
                    if found_in_bin:
                        break
                
                if found_in_bin:
                    break
            
            if best_bin >= 0 and best_pos is not None:
                x, y, z = best_pos
                bins[best_bin].append((x, y, z, w, h, d))
                placement_map[idx] = (best_bin, x, y, z, w, h, d)
                bin_volumes[best_bin] += w * h * d
            else:
                # Open a new bin
                new_bin = len(bins)
                bins.append([])
                bins[new_bin].append((0, 0, 0, w, h, d))
                placement_map[idx] = (new_bin, 0, 0, 0, w, h, d)
                bin_volumes.append(w * h * d)
            
            # Check deadline
            if time.time() > deadline:
                break
        
        # Handle any unplaced boxes
        for idx in range(n):
            if placement_map[idx] is None:
                new_bin = len(bins)
                bins.append([])
                w, h, d = boxes[idx]
                bins[new_bin].append((0, 0, 0, w, h, d))
                placement_map[idx] = (new_bin, 0, 0, 0, w, h, d)
                if new_bin < len(bin_volumes):
                    bin_volumes[new_bin] += w * h * d
                else:
                    bin_volumes.append(w * h * d)
        
        num_bins = len(bins)
        # Compute last-bin fill fraction
        if num_bins > 0:
            last_bin = max(placement_map[i][0] for i in range(n) if placement_map[i] is not None)
            last_fill = bin_volumes[last_bin] / (W * H * D) if W * H * D > 0 else 0.0
        else:
            last_fill = 0.0
        
        return num_bins, placement_map, last_fill
    
    # Define sorting strategies
    volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    max_dims = [max(boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    base_areas = [boxes[i][0] * boxes[i][1] for i in range(n)]
    
    strategies = [
        sorted(range(n), key=lambda i: -volumes[i]),           # volume desc
        sorted(range(n), key=lambda i: (-max_dims[i], -volumes[i])),  # max dim, then volume
        sorted(range(n), key=lambda i: (-base_areas[i], -volumes[i])), # base area, then volume
        sorted(range(n), key=lambda i: (-volumes[i], -max_dims[i])),   # volume, then max dim
    ]
    
    best_result = None
    best_score_val = float('inf')
    
    for strategy in strategies:
        if time.time() > deadline:
            break
        
        num_bins, placement_map, last_fill = solve_with_strategy(strategy)
        
        # Score: B + 0.1 * (1 - last_bin_fill)
        score_val = num_bins + 0.1 * (1 - last_fill)
        
        if score_val < best_score_val:
            best_score_val = score_val
            best_result = placement_map
    
    # If no strategy completed, use a fallback
    if best_result is None:
        # Simple fallback: one box per bin
        best_result = []
        for i in range(n):
            w, h, d = boxes[i]
            best_result.append((i, 0, 0, 0, w, h, d))
    
    # Build result in original order
    placements = []
    for idx in range(n):
        b, x, y, z, w, h, d = best_result[idx]
        placements.append((b, x, y, z, w, h, d))
    
    return placements
# EVOLVE-BLOCK-END
