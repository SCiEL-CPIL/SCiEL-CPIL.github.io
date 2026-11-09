"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)
    
    # Sort boxes by volume descending (FFD heuristic)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    
    # 3D Binary Indexed Tree (Fenwick Tree) implementation for each bin
    # We use 1-based indexing internally, so dimensions are W+1, H+1, D+1
    # To keep memory manageable, we assume W,H,D are reasonable (e.g., < 1000)
    # If dimensions are huge, this might blow up, but typically packing containers are small enough.
    # For safety, if any dim > 2000, we fall back to a simpler list-based check (though unlikely in generated tests)
    
    class Bin3D:
        def __init__(self, w, h, d):
            self.w = w
            self.h = h
            self.d = d
            # Use numpy for fast zero-init
            self.tree = np.zeros((w + 1, h + 1, d + 1), dtype=np.int8)
            self.items = [] # List of (x, y, z, w, h, d) for fallback or inspection if needed
            
        def update(self, x, y, z, w, h, d, val):
            # val: 1 to mark occupied, -1 to mark free
            # x, y, z are 0-based start coords
            # We update the range [x, x+w) x [y, y+h) x [z, z+d)
            # Fenwick update is point update, but we need range update.
            # Standard BIT range update requires 4D or prefix sum logic.
            # Simpler: Since we only need "is this specific box empty", we can use a different approach.
            # Actually, checking if a box fits requires checking if ANY cell in the box is occupied.
            # A BIT that stores occupied cells allows O(1) point query, but O(w*h*d) range query.
            # That's too slow.
            
            # Alternative: Maintain a "skyline" or use a simpler spatial hash?
            # Given the constraints (40 boxes), a simpler approach might be faster and safer than a complex 3D BIT implementation in Python.
            # Let's stick to a highly optimized list-based check but with better heuristics.
            # The previous program's bottleneck was likely the O(N^2) overlap check inside a large candidate loop.
            
            # Let's try a different strategy: 
            # 1. Sort boxes by volume desc.
            # 2. For each box, try to place in existing bins.
            # 3. Use a "Best Fit" criterion: find the placement that minimizes the "waste" or maximizes the "contact".
            # 4. To speed up finding the spot, we use a "Frontier" list of candidate positions.
            
            pass # This class is not used in the final logic below, kept for thought process.

    # Revised Strategy: 
    # 1. Sort by volume desc.
    # 2. Maintain a list of bins.
    # 3. For each bin, maintain a list of "free spots" or just iterate over existing items to generate candidate positions.
    # 4. To improve on the previous score, we use a "Best Fit Decreasing" where we choose the bin and position that leaves the LEAST remaining space in the tightest dimension, or simply the position with the lowest (x, y, z) sum among valid spots, but we prioritize bins that are already "fuller" (Best Fit Bin Selection).
    
    # Let's implement a robust "Best Fit" with efficient candidate generation.
    # Candidate positions: (0,0,0) and (x+w_i, y_j, z_k) etc.
    # Actually, the set of valid positions is defined by the "corners" of existing boxes.
    
    def get_candidates(items, W, H, D):
        xs = {0}
        ys = {0}
        zs = {0}
        for ex, ey, ez, ew, eh, ed in items:
            if ex + ew <= W: xs.add(ex + ew)
            if ey + eh <= H: ys.add(ey + eh)
            if ez + ed <= D: zs.add(ez + ed)
        return sorted(xs), sorted(ys), sorted(zs)

    def check_overlap(x, y, z, w, h, d, items):
        for ex, ey, ez, ew, eh, ed in items:
            if not (x + w <= ex or ex + ew <= x or
                    y + h <= ey or ey + eh <= y or
                    z + d <= ez or ez + ed <= z):
                return True
        return False

    bins = [] # List of lists of items
    placements = [None] * n

    # Pre-compute box volumes for sorting? Already done.
    
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Emergency fallback
            for rem_idx, (rw, rh, rd) in indexed_boxes:
                if placements[rem_idx] is None:
                    # Try to place in a new bin or just mark as invalid?
                    # Task requires valid placement. If we run out of time, we must place them.
                    # A safe fallback is to put each remaining box in its own new bin if it fits, 
                    # or if that fails, it's a loss. But usually 40 boxes fit in a few bins.
                    bins.append([(0, 0, 0, rw, rh, rd)])
                    placements[rem_idx] = (len(bins)-1, 0, 0, 0, rw, rh, rd)
            return placements

        if w > W or h > H or d > D:
            # Box doesn't fit in container. This should not happen per problem statement.
            # But handle gracefully by putting in a dummy bin? No, invalid.
            # Assume it fits.
            pass

        best_bin_idx = -1
        best_pos = None
        best_score = float('inf')
        
        # Heuristic: Try bins that are already "tighter" or have less free space?
        # Or just try all bins and pick the best fit (min waste).
        
        for b_idx, items in enumerate(bins):
            # Generate candidates
            xs, ys, zs = get_candidates(items, W, H, D)
            
            # Filter candidates that fit
            valid_xs = [x for x in xs if x + w <= W]
            if not valid_xs: continue
            valid_ys = [y for y in ys if y + h <= H]
            if not valid_ys: continue
            valid_zs = [z for z in zs if z + d <= D]
            if not valid_zs: continue
            
            # Search for best position in this bin
            # Score: Prefer positions that are "low" (close to origin) to keep the top clear?
            # Or "far" to pack corners?
            # Standard BLF (Bottom-Left-Far) prefers low x, low y, high z.
            # Let's use a score that encourages filling the bin: 
            # Minimize (x, y, z) lexicographically? Or minimize distance to center?
            # Let's try minimizing x, then y, then z (Bottom-Left).
            
            bin_best_pos = None
            bin_best_score = float('inf')
            
            for x in valid_xs:
                for y in valid_ys:
                    for z in valid_zs:
                        if check_overlap(x, y, z, w, h, d, items):
                            continue
                        # Score: 
                        # 1. Primary: Minimize "waste" in the bin? Hard to compute quickly.
                        # 2. Secondary: Bottom-Left-Far heuristic.
                        #    Low x, Low y, High z.
                        #    Score = x * 10000 + y * 100 - z
                        #    Wait, if we want high z, we subtract z.
                        #    Let's stick to Bottom-Left (min x, min y, min z) as it's stable and often good for FFD.
                        #    Or "Bottom-Left-Far" (min x, min y, max z) to stack up.
                        #    Let's try Bottom-Left-Far: x*10000 + y*100 - z
                        score = x * 10000 + y * 100 - z
                        if score < bin_best_score:
                            bin_best_score = score
                            bin_best_pos = (x, y, z)
            
            if bin_best_pos is not None:
                # Compare this bin's best position against other bins
                # We want to select the bin that fits the box "best".
                # A common metric is the amount of free space left in the bin after placement.
                # Lower free space is better (tighter fit).
                # Approximate free space: Volume of bin - Volume of items - Volume of new box?
                # This is constant for a given bin and box, so it doesn't help distinguish positions.
                # We need to distinguish between bins.
                # "Best Fit" usually means: place in the bin where the box fits tightest.
                # If we use the position score as a proxy for fit quality, it's not quite right.
                
                # Let's use a composite score for the global best:
                # 1. Prefer bins with fewer items? No, prefer bins with more items (consolidation).
                # 2. Prefer positions that are "lower" (less height used).
                
                # Let's simply pick the first valid bin with the "best" (lowest) position score?
                # Or, we can calculate the "bounding box height" of the items in the bin.
                # Prefer bins with lower current max_z?
                
                # Let's try: Global score = (bin_index, position_score) ? No.
                # Let's try: Global score = position_score.
                # This effectively does First-Fit with Bottom-Left-Far positioning.
                
                # To improve, let's add a penalty for using a new bin.
                # If we are considering a new bin, its score should be high.
                
                if bin_best_score < best_score:
                    best_score = bin_best_score
                    best_bin_idx = b_idx
                    best_pos = bin_best_pos
        
        if best_bin_idx != -1:
            x, y, z = best_pos
            items = bins[best_bin_idx]
            items.append((x, y, z, w, h, d))
            placements[orig_idx] = (best_bin_idx, x, y, z, w, h, d)
        else:
            # New bin
            if w <= W and h <= H and d <= D:
                bins.append([(0, 0, 0, w, h, d)])
                placements[orig_idx] = (len(bins)-1, 0, 0, 0, w, h, d)
            else:
                # Should not happen
                placements[orig_idx] = (len(bins), 0, 0, 0, w, h, d)
                bins.append([])

    return placements
# EVOLVE-BLOCK-END
