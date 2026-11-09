"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    W, H, D = container
    container_vol = W * H * D
    
    # Sort boxes by volume descending (FFD/BFD)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    
    # Bins: list of dicts or structures holding occupancy and items
    # We will maintain a grid for each bin for fast overlap checking
    # Grid size W x H x D is potentially large, but for 40 boxes and typical container sizes 
    # in these benchmarks (often 100x100x100 or similar), it's manageable.
    # However, to be safe against memory/time if container is huge, we limit grid usage 
    # or use a list of occupied boxes for small item counts.
    # Given 40 boxes, a list-based check is O(N) per candidate.
    # A grid check is O(1) per candidate but setup is O(W*H*D).
    # Let's use a hybrid: if W*H*D < 10^6, use grid. Else use list.
    
    # Actually, for 3D bin packing with 40 boxes, the number of candidate positions is small.
    # The bottleneck is usually the search for the best position.
    # Let's stick to a robust list-based approach with a smart candidate generator 
    # and a "Best Fit" metric that prefers positions that "tuck" items together.
    
    bins = [] # List of lists of (x, y, z, w, h, d)
    bin_volumes = [] # Total volume in each bin
    
    def get_candidate_positions(bin_items, w, h, d):
        """
        Generate candidate (x, y, z) positions for a box of size (w, h, d)
        in a bin with existing items.
        Candidates are derived from the "corners" of existing items.
        """
        xs = {0}
        ys = {0}
        zs = {0}
        
        for ex, ey, ez, ew, eh, ed in bin_items:
            # Right side
            if ex + ew + w <= W:
                xs.add(ex + ew)
            # Top side
            if ey + eh + h <= H:
                ys.add(ey + eh)
            # Back side
            if ez + ed + d <= D:
                zs.add(ez + ed)
            
            # Also consider positions aligned with the "left/bottom/front" of existing items?
            # No, we only place in empty space. The "corners" of existing items define the 
            # boundaries of free space in a greedy packing.
            
        # Filter candidates that fit
        valid_xs = [x for x in xs if x + w <= W]
        valid_ys = [y for y in ys if y + h <= H]
        valid_zs = [z for z in zs if z + d <= D]
        
        # To avoid explosion, we limit the number of candidates if there are too many.
        # With 40 boxes, max items in a bin is 40. 
        # Max xs ~ 41, ys ~ 41, zs ~ 41. Total candidates ~ 41^3 = 68921.
        # This is acceptable for 40 boxes * 41^3 checks? 
        # 40 * 68921 * 40 (overlap check) = ~110 million operations. Might be too slow in Python.
        
        # Optimization: Check overlap early.
        # We will iterate through candidates and check overlap.
        
        return valid_xs, valid_ys, valid_zs

    def check_overlap(x, y, z, w, h, d, existing):
        for ex, ey, ez, ew, eh, ed in existing:
            if not (x + w <= ex or ex + ew <= x or
                    y + h <= ey or ey + eh <= y or
                    z + d <= ez or ez + ed <= z):
                return True
        return False

    def find_best_position(bin_items, w, h, d, bin_vol):
        """
        Find the position that minimizes 'waste' or maximizes 'tightness'.
        We define a score for a position. Lower is better.
        Score components:
        1. Distance to origin (prefers bottom-left-front)
        2. Contact with existing items (prefers tucking in)
        
        Let's use a simple heuristic: Minimize (x, y, z) lexicographically 
        among valid positions, but with a twist: 
        We prefer positions that are "supported" or "tucked".
        
        Actually, a common BFD metric is: 
        For each valid position, calculate the "remaining free volume" in the bin 
        if this box is placed there. Choose the position that leaves the least free volume?
        That's hard to calculate exactly without a 3D grid.
        
        Alternative: 
        Minimize the sum of distances to the "center of mass" of the bin?
        Or simply: Minimize x, then y, then z. This is First Fit.
        
        To make it Best Fit, we can look for the position that minimizes the 
        "empty space" created. 
        A proxy: Minimize the maximum distance to any existing item's face?
        
        Let's try a simpler, effective heuristic:
        For each valid position, calculate a "cost".
        Cost = x + y + z (prefers corner)
        BUT, we also want to fill gaps.
        
        Let's use a 3D grid for the *current bin* only to check overlaps and 
        find the "best" spot based on a heuristic like "minimize exposed surface area"
        or simply "minimize x, then y, then z" but scanning all valid positions 
        and picking the one with the smallest (x, y, z) tuple.
        
        Wait, the current program already does something similar but it's First Fit 
        (breaks on first bin that fits). 
        The key difference for BFD is: Check ALL bins, find ALL valid positions in ALL bins, 
        and pick the global best position across all bins.
        
        What is the "best" position?
        Usually, minimizing the "waste" is good. 
        A simple proxy for waste in 3D packing is the "distance to the corner" 
        (x, y, z). Smaller x,y,z is generally better for packing density.
        
        So, BFD strategy:
        1. For each box (sorted by size desc):
           a. For each existing bin:
              - Find all valid (x,y,z) positions.
              - For each valid position, calculate a score.
              - Keep the best score for this bin.
           b. If no existing bin has a valid position, open a new bin at (0,0,0).
           c. Place the box in the bin/position with the globally best score.
        
        Score function: 
        We want to minimize the "empty space" left behind. 
        A good heuristic is to minimize the "potential" of the position.
        Let's use: score = x*10000 + y*100 + z. (Prefers bottom-left-front)
        This is essentially First Fit Decreasing but checking all bins for the 
        absolute "lowest" position available.
        
        Is this better than First Fit? 
        First Fit puts the box in the *first* bin it fits. 
        Best Fit puts the box in the *tightest* spot available across all bins.
        This often leads to fewer bins because it packs items more tightly together, 
        leaving larger contiguous free spaces in other bins (which are harder to use) 
        vs. smaller, more fragmented spaces? 
        Actually, Best Fit usually minimizes the *remaining* free space in the chosen bin.
        
        Let's implement a score that reflects "how much room is left".
        Since we can't easily calc remaining free space, we use the "corner" heuristic.
        Placing at (0,0,0) is best. Placing at (10, 0, 0) is worse.
        
        To improve on simple corner heuristic, we can add a penalty for "gaps".
        But let's start with the standard BFD: 
        Find the position that minimizes (x, y, z) across all bins.
        """
        best_score = float('inf')
        best_pos = None
        
        valid_xs, valid_ys, valid_zs = get_candidate_positions(bin_items, w, h, d)
        
        # Optimization: If there are too many candidates, sample or prune.
        # For now, iterate all.
        
        # We can prune xs, ys, zs if they are too large.
        # But with 40 boxes, it's likely fine.
        
        # Sort candidates to check "best" first? 
        # If we sort xs, ys, zs ascending, the first valid one we find 
        # with a specific tie-breaking rule might be good.
        # But we need to check ALL to find the global min (x,y,z).
        
        # Actually, since score is monotonic with x, y, z, 
        # we can just iterate x, then y, then z in ascending order, 
        # and the FIRST valid position we find is the one with min x, 
        # then min y for that x, then min z for that x,y.
        # This is the "most bottom-left-front" position.
        
        # So we just need the first valid position in sorted order.
        
        for x in sorted(valid_xs):
            for y in sorted(valid_ys):
                for z in sorted(valid_zs):
                    if not check_overlap(x, y, z, w, h, d, bin_items):
                        # Found the best position for this bin (min x,y,z)
                        # Score for this bin is this position.
                        # We return this position and its "score" 
                        # so the caller can compare across bins.
                        # Score: x*W*H + y*W + z (linear index)
                        return (x, y, z)
        
        return None

    placements = [None] * len(boxes)
    
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Emergency fallback
            for rem_idx, (rw, rh, rd) in indexed_boxes:
                if placements[rem_idx] is None:
                    # Place in new bin
                    bin_idx = len(bins)
                    bins.append([(0, 0, 0, rw, rh, rd)])
                    placements[rem_idx] = (bin_idx, 0, 0, 0, rw, rh, rd)
            return placements

        best_bin_idx = -1
        best_pos = None
        best_score = float('inf')
        
        # Check all existing bins
        for b_idx in range(len(bins)):
            pos = find_best_position(bins[b_idx], w, h, d, bin_volumes[b_idx])
            if pos is not None:
                x, y, z = pos
                # Score: prefer lower x, then y, then z
                # Using linear index to compare across bins
                score = x * H * D + y * D + z
                if score < best_score:
                    best_score = score
                    best_pos = pos
                    best_bin_idx = b_idx
        
        if best_bin_idx != -1:
            x, y, z = best_pos
            bins[best_bin_idx].append((x, y, z, w, h, d))
            placements[orig_idx] = (best_bin_idx, x, y, z, w, h, d)
        else:
            # Open new bin
            if w <= W and h <= H and d <= D:
                bin_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placements[orig_idx] = (bin_idx, 0, 0, 0, w, h, d)
            else:
                # Should not happen
                bin_idx = len(bins)
                bins.append([])
                placements[orig_idx] = (bin_idx, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
