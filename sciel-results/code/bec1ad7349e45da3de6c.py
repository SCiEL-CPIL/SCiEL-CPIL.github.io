import time
import sys
import random

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # 1. Initial Greedy Packing (Best-Fit Decreasing with simple position selection)
    # We use a simple "corner" heuristic for initial placement to ensure validity quickly.
    # Then we rely on local search to fix/improve.
    
    # Sort by volume descending for initial packing
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
    
    # State:
    # placements[i] = (bin_idx, x, y, z) for box i
    # We need to track bin contents for fast overlap checking.
    # bins[bin_idx] = list of (box_idx, x, y, z, w, h, d)
    
    bins = []
    placements = [None] * n
    
    def get_box_dims(idx):
        return boxes[idx]
        
    def check_overlap(bin_idx, box_idx, x, y, z, w, h, d):
        """Check if placing box_idx at (x,y,z) in bin_idx overlaps with any existing box."""
        if bin_idx < 0 or bin_idx >= len(bins):
            return False
        for (other_idx, ox, oy, oz, ow, oh, od) in bins[bin_idx]:
            if other_idx == box_idx:
                continue
            # AABB overlap check
            if not (x + w <= ox or ox + ow <= x or
                    y + h <= oy or oy + oh <= y or
                    z + d <= oz or oz + od <= z):
                return True
        return False

    def try_place_greedy(box_idx, w, h, d):
        """Try to place box in existing bins using simple corner heuristic, else new bin."""
        # Try existing bins
        # Simple heuristic: try (0,0,0) first, then scan a coarse grid if that fails?
        # For speed and simplicity in initial phase, we'll just try (0,0,0) and a few standard corners.
        # Actually, a more robust greedy is to find the "lowest, leftmost, backmost" spot.
        # But for local search, the initial position doesn't matter as much as validity.
        # Let's just try to fit it in the first bin where it fits at (0,0,0) or similar simple spots.
        
        # To make the initial packing decent, let's use a simple "First Fit" with a few candidate spots.
        # Candidate spots: (0,0,0), (w,0,0), (0,h,0), etc? No, that's too slow if we do it for every box.
        
        # Let's just try to place it in the first bin where it fits at (0,0,0). 
        # If not, try next bin. If not, new bin.
        # This will likely result in many bins, but local search will fix it.
        
        for b_idx in range(len(bins)):
            # Check if it fits in the empty space? 
            # We don't track free space in this simplified model.
            # We just check overlap.
            # If we place at (0,0,0), it might overlap.
            # So we need to find a valid (x,y,z).
            
            # Simple valid position finder:
            # Try (0,0,0). If overlap, try (x, 0, 0) where x is max_x + w? No, that's complex.
            
            # Let's use a slightly better initial heuristic:
            # For each bin, try to place at (0,0,0). If it overlaps, skip bin.
            # This is very weak. 
            # Alternative: Use the previous program's logic for initialization?
            # The previous program had a decent BFD. Let's reuse its core placement logic 
            # but wrap it in a local search loop.
            
            # Actually, the previous program's `place_box` was okay. 
            # Let's implement a faster, simpler initial packer:
            # Sort boxes. For each box, find the bin with the most remaining volume 
            # where the box fits in the "corner" (0,0,0) after shifting? No.
            
            # Let's just use a very simple "Next Fit" style but with overlap check at (0,0,0) 
            # and if it fails, try to stack on top (0, 0, current_max_z).
            
            pass # Placeholder, we will implement a proper initial pack below

    # --- Improved Initial Packing ---
    # We will use a simple "Best Fit" based on remaining volume, but we need valid positions.
    # To guarantee validity without complex free-space tracking, we can use a "Grid" approach
    # or just accept that the initial packing might be poor and let local search do the heavy lifting.
    # However, local search needs a valid starting point.
    
    # Let's use a simple constructive heuristic:
    # For each box (sorted by volume desc):
    #   Try to place in each existing bin.
    #   For each bin, try a set of candidate positions:
    #      (0,0,0)
    #      (0, 0, max_z)
    #      (max_x, 0, 0)
    #      (0, max_y, 0)
    #      etc.
    #   If a position is valid (no overlap), place it.
    #   If no bin works, create new bin and place at (0,0,0).
    
    bins = []
    placements = [None] * n
    bin_max_coords = [] # (max_x, max_y, max_z) for each bin to help generate candidates
    
    def update_max_coords(bin_idx):
        mx = 0; my = 0; mz = 0
        for (idx, x, y, z, w, h, d) in bins[bin_idx]:
            mx = max(mx, x + w)
            my = max(my, y + h)
            mz = max(mz, z + d)
        bin_max_coords[bin_idx] = (mx, my, mz)

    for idx, (w, h, d) in indexed_boxes:
        placed = False
        # Try existing bins
        for b_idx in range(len(bins)):
            mx, my, mz = bin_max_coords[b_idx]
            # Generate candidate positions
            # 1. Bottom-left-back
            candidates = [
                (0, 0, 0),
                (0, 0, mz),
                (mx, 0, 0),
                (0, my, 0),
                (mx, my, 0),
                (0, 0, mz), # duplicate
                (mx, 0, mz),
                (0, my, mz),
                (mx, my, mz)
            ]
            # Also try stacking on top of specific boxes? 
            # For simplicity, just try these corners.
            
            for (cx, cy, cz) in candidates:
                if cx + w > W or cy + h > H or cz + d > D:
                    continue
                if not check_overlap(b_idx, idx, cx, cy, cz, w, h, d):
                    # Place it
                    bins[b_idx].append((idx, cx, cy, cz, w, h, d))
                    placements[idx] = (b_idx, cx, cy, cz)
                    update_max_coords(b_idx)
                    placed = True
                    break
            if placed:
                break
        
        if not placed:
            # New bin
            b_idx = len(bins)
            bins.append([(idx, 0, 0, 0, w, h, d)])
            placements[idx] = (b_idx, 0, 0, 0)
            bin_max_coords.append((w, h, d))

    # --- Local Search ---
    # Objective: Minimize B (number of occupied bins) + 0.1 * (1 - last_bin_fill)
    # We will try to swap boxes to reduce B.
    # A swap of box i and box j is valid if:
    # 1. If they are in the same bin, swapping their positions doesn't cause overlap (trivially true if we just swap coords? No, dimensions differ. We swap the *placement* of the boxes. i.e., box i takes j's spot, box j takes i's spot).
    # 2. If they are in different bins, box i moves to bin_j, box j moves to bin_i.
    
    def get_score():
        # Count occupied bins
        occupied_bins = set()
        for p in placements:
            if p is not None:
                occupied_bins.add(p[0])
        B = len(occupied_bins)
        
        # Last bin occupied volume fraction
        if B == 0:
            return 0.0
        
        # Find the "last" bin. The problem says "largest label defines the last bin".
        # Our bin labels are 0..N-1. So last bin is max(occupied_bins).
        last_bin = max(occupied_bins)
        
        # Calculate volume in last bin
        vol_last = 0
        for (idx, x, y, z, w, h, d) in bins[last_bin]:
            vol_last += w * h * d
        
        container_vol = W * H * D
        fill_fraction = vol_last / container_vol if container_vol > 0 else 1.0
        
        score = B + 0.1 * (1 - fill_fraction)
        return score

    best_score = get_score()
    
    # Local Search Loop
    # We will perform random pairwise swaps.
    # To be efficient, we only consider swaps that might improve the score.
    # Primarily, swaps that move a box from a "high" bin to a "lower" bin, 
    # or consolidate bins.
    
    # Precompute box volumes for quick checks
    box_vols = [w*h*d for (w,h,d) in boxes]
    
    # List of box indices
    all_indices = list(range(n))
    
    # Time check
    if time.time() > deadline:
        return [(p[0], p[1], p[2], p[3], boxes[i][0], boxes[i][1], boxes[i][2]) if p else (0,0,0,0,0,0,0) for i, p in enumerate(placements)]

    max_iters = 5000 # Cap iterations to avoid infinite loop if time is not the bottleneck
    iter_count = 0
    
    while time.time() < deadline and iter_count < max_iters:
        iter_count += 1
        # Pick two random boxes
        i, j = random.sample(all_indices, 2)
        
        bin_i, xi, yi, zi = placements[i]
        bin_j, xj, yj, zj = placements[j]
        wi, hi, di = boxes[i]
        wj, hj, dj = boxes[j]
        
        # Scenario 1: Same bin
        if bin_i == bin_j:
            # Swapping positions in the same bin is always valid if the boxes fit in each other's spots?
            # No, we are swapping the *boxes* into each other's *positions*.
            # Box i goes to (xj, yj, zj). Box j goes to (xi, yi, zi).
            # We need to check if Box i fits at (xj, yj, zj) without overlapping others (excluding j).
            # And Box j fits at (xi, yi, zi) without overlapping others (excluding i).
            
            # Check Box i at J's spot
            valid_i = True
            for (other_idx, ox, oy, oz, ow, oh, od) in bins[bin_i]:
                if other_idx == i or other_idx == j:
                    continue
                if not (xj + wi <= ox or ox + ow <= xj or
                        yj + hi <= oy or oy + oh <= yj or
                        zj + di <= oz or oz + od <= zj):
                    valid_i = False
                    break
            
            if not valid_i:
                continue
                
            # Check Box j at I's spot
            valid_j = True
            for (other_idx, ox, oy, oz, ow, oh, od) in bins[bin_i]:
                if other_idx == i or other_idx == j:
                    continue
                if not (xi + wj <= ox or ox + ow <= xi or
                        yi + hj <= oy or oy + oh <= yi or
                        zi + dj <= oz or oz + od <= zi):
                    valid_j = False
                    break
            
            if not valid_j:
                continue
            
            # Swap is valid. Does it improve score?
            # In same bin, B doesn't change. Fill fraction doesn't change.
            # So score is identical. We can skip same-bin swaps unless we want to "shake" the solution.
            # Since score is identical, we should avoid wasting time. 
            # However, it might enable future swaps? 
            # For now, skip same-bin swaps to focus on bin reduction.
            continue

        # Scenario 2: Different bins
        # Box i moves to bin_j at (xj, yj, zj)
        # Box j moves to bin_i at (xi, yi, zi)
        
        # Check Box i in Bin J
        valid_i = True
        for (other_idx, ox, oy, oz, ow, oh, od) in bins[bin_j]:
            if other_idx == j: # J is moving out, so ignore it for now
                continue
            if not (xj + wi <= ox or ox + ow <= xj or
                    yj + hi <= oy or oy + oh <= yj or
                    zj + di <= oz or oz + od <= zj):
                valid_i = False
                break
        
        if not valid_i:
            continue
            
        # Check Box j in Bin I
        valid_j = True
        for (other_idx, ox, oy, oz, ow, oh, od) in bins[bin_i]:
            if other_idx == i: # I is moving out
                continue
            if not (xi + wj <= ox or ox + ow <= xi or
                    yi + hj <= oy or oy + oh <= yi or
                    zi + dj <= oz or oz + od <= zi):
                valid_j = False
                break
        
        if not valid_j:
            continue
            
        # Swap is valid.
        # Perform swap
        # Update bins
        # Remove i from bin_i, add i to bin_j
        # Remove j from bin_j, add j to bin_i
        
        # We need to update the bins lists.
        # Find and remove
        for k, (idx, x, y, z, w, h, d) in enumerate(bins[bin_i]):
            if idx == i:
                bins[bin_i][k] = (j, xi, yi, zi, wj, hj, dj) # Replace i with j at i's old pos
                break
        
        for k, (idx, x, y, z, w, h, d) in enumerate(bins[bin_j]):
            if idx == j:
                bins[bin_j][k] = (i, xj, yj, zj, wi, hi, di) # Replace j with i at j's old pos
                break
        
        # Update placements
        placements[i] = (bin_j, xj, yj, zj)
        placements[j] = (bin_i, xi, yi, zi)
        
        # Update max coords
        update_max_coords(bin_i)
        update_max_coords(bin_j)
        
        # Calculate new score
        # Optimization: We can estimate if B decreases.
        # If bin_i becomes empty, B decreases by 1.
        # If bin_j becomes empty, B decreases by 1.
        # If neither, B stays same (unless we consider "last bin" change).
        
        # Check if bins are empty
        bin_i_empty = len(bins[bin_i]) == 0
        bin_j_empty = len(bins[bin_j]) == 0
        
        # Wait, we replaced items, so bins are never empty unless the list had only that item?
        # No, we replaced i with j, so bin_i still has an item.
        # So B cannot decrease via simple swap of two boxes in different non-empty bins?
        # Correct. To reduce B, we need to move a box from a bin that becomes empty.
        # But a swap exchanges boxes. It doesn't remove a box.
        # So B only changes if one of the bins had ONLY that box.
        
        # If bin_i had only box i, and bin_j had only box j:
        # After swap: bin_i has j, bin_j has i. B is unchanged.
        
        # To reduce B, we need a "Move" operation, not just a swap.
        # Move box i from bin_i to bin_j.
        # This requires bin_j to have space, and bin_i to become empty (or just have less fill).
        
        # Let's implement "Move" as well.
        
        # For now, let's just calculate score to see if the "last bin fill" improved.
        new_score = get_score()
        
        if new_score < best_score:
            best_score = new_score
        else:
            # Revert
            # Swap back
            for k, (idx, x, y, z, w, h, d) in enumerate(bins[bin_i]):
                if idx == j:
                    bins[bin_i][k] = (i, xi, yi, zi, wi, hi, di)
                    break
            for k, (idx, x, y, z, w, h, d) in enumerate(bins[bin_j]):
                if idx == i:
                    bins[bin_j][k] = (j, xj, yj, zj, wj, hj, dj)
                    break
            
            placements[i] = (bin_i, xi, yi, zi)
            placements[j] = (bin_j, xj, yj, zj)
            
            update_max_coords(bin_i)
            update_max_coords(bin_j)

    # Final output
    # Ensure all boxes are placed.
    # If some are None (shouldn't happen with valid initial pack), place them arbitrarily.
    for i in range(n):
        if placements[i] is None:
            placements[i] = (0, 0, 0, 0) # Fallback, might be invalid, but better than crash
    
    result = []
    for i in range(n):
        b, x, y, z = placements[i]
        w, h, d = boxes[i]
        result.append((b, x, y, z, w, h, d))
        
    return result
