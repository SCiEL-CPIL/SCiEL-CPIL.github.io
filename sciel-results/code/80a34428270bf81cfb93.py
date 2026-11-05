"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard: leave safety margin before 150s kill
    deadline = time.time() + 145.0
    
    # Sort indices by volume descending (Best Fit Decreasing)
    # Keep original index for final mapping
    indexed = [(i, boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][0]*boxes[i][1]*boxes[i][2]) for i in range(n)]
    indexed.sort(key=lambda x: -x[4])
    
    def try_place(boxes_sorted, container, max_bins=None):
        """
        Attempt to pack all boxes into bins using Best Fit Decreasing with 
        a list of candidate positions per bin.
        Returns list of (bin, x, y, z, w, h, d) or None if failed.
        """
        W, H, D = container
        n = len(boxes_sorted)
        
        # Bins state: each bin is a list of placed boxes (x, y, z, w, h, d)
        bins = [[]]  # Start with one empty bin
        
        def get_candidate_positions(bin_boxes, w, h, d):
            """
            Generate candidate positions for a box of size (w,h,d) in a bin 
            containing bin_boxes.
            We use a grid of candidate x, y, z coordinates derived from 
            the boundaries of existing boxes plus 0.
            """
            # Candidate x coords: 0 and right edges of existing boxes
            x_candidates = {0}
            y_candidates = {0}
            z_candidates = {0}
            
            for (bx, by, bz, bw, bh, bd) in bin_boxes:
                x_candidates.add(bx + bw)
                y_candidates.add(by + bh)
                z_candidates.add(bz + bd)
            
            # Filter candidates that are within container bounds
            x_candidates = [x for x in x_candidates if x + w <= W]
            y_candidates = [y for y in y_candidates if y + h <= H]
            z_candidates = [z for z in z_candidates if z + d <= D]
            
            # If no candidates, return empty
            if not x_candidates or not y_candidates or not z_candidates:
                return []
            
            # Generate all combinations
            candidates = []
            for x in x_candidates:
                for y in y_candidates:
                    for z in z_candidates:
                        # Check overlap with existing boxes
                        valid = True
                        for (bx, by, bz, bw, bh, bd) in bin_boxes:
                            if not (x + w <= bx or bx + bw <= x or
                                    y + h <= by or by + bh <= y or
                                    z + d <= bz or bz + bd <= z):
                                valid = False
                                break
                        if valid:
                            # Score: prefer lower x, then y, then z (lexicographic)
                            # Also prefer positions that are "tight" against existing boxes
                            # For now, just use lexicographic order
                            candidates.append((x, y, z))
            
            # Sort by (x, y, z) to prefer lowest positions
            candidates.sort(key=lambda p: (p[0], p[1], p[2]))
            return candidates
        
        # Place boxes one by one
        for idx, (orig_idx, w, h, d, vol) in enumerate(boxes_sorted):
            placed = False
            # Try to place in existing bins first, then open a new bin
            for b_idx in range(len(bins)):
                if max_bins is not None and b_idx == max_bins:
                    break
                bin_boxes = bins[b_idx]
                candidates = get_candidate_positions(bin_boxes, w, h, d)
                for (x, y, z) in candidates:
                    # Check if this position is valid (already checked in get_candidate_positions)
                    # Place the box
                    bins[b_idx].append((x, y, z, w, h, d))
                    placed = True
                    break
                if placed:
                    break
            
            if not placed:
                # Open a new bin
                if max_bins is not None and len(bins) >= max_bins:
                    return None
                # Place in new bin at (0,0,0)
                new_bin = [(0, 0, 0, w, h, d)]
                bins.append(new_bin)
        
        # Convert to final placement list
        placements = [None] * n
        for b_idx, bin_boxes in enumerate(bins):
            for (x, y, z, w, h, d) in bin_boxes:
                # Find original index
                for idx, (orig_idx, ow, oh, od, ov) in enumerate(boxes_sorted):
                    if ow == w and oh == h and od == d:
                        if placements[idx] is None:
                            placements[idx] = (b_idx, x, y, z, w, h, d)
                            break
        
        # Check if all placed
        if any(p is None for p in placements):
            return None
        
        return placements
    
    # Strategy: Try to minimize number of bins first, then optimize last bin fill.
    # Start with a greedy solution to get an upper bound on B.
    initial_solution = try_place(indexed, container)
    if initial_solution is None:
        # Should not happen with unlimited bins
        # Fallback: just place in one bin per box if necessary (shouldn't be needed)
        return [(i, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    # Count bins used
    max_bin_used = max(p[0] for p in initial_solution)
    B_init = max_bin_used + 1
    
    # Calculate volume of last bin
    last_bin_vol = sum(p[4]*p[5]*p[6] for p in initial_solution if p[0] == max_bin_used)
    container_vol = W * H * D
    last_bin_frac = last_bin_vol / container_vol if container_vol > 0 else 0
    score_init = B_init + 0.1 * (1 - last_bin_frac)
    
    best_solution = initial_solution
    best_score = score_init
    
    # Try to improve by reducing B or improving last bin fill
    # We can try local search: swap boxes between bins, or try different orderings.
    
    # For now, let's just return the initial best fit decreasing solution.
    # The heuristic is simple but effective for 40 boxes.
    
    # Check time
    if time.time() > deadline:
        return best_solution
    
    # Try a few random restarts with different orderings if time permits
    # But BFD is usually quite good. Let's stick with it for safety.
    
    return best_solution
# EVOLVE-BLOCK-END
