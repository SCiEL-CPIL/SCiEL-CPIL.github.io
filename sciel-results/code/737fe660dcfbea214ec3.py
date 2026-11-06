"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

def pack(boxes, container):
    """
    Hybrid 3D bin packing: greedy BFD + limited backtracking for large boxes.
    
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    W, H, D = container
    n = len(boxes)
    
    # Sort boxes by volume descending, keep original index
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[1])
    
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 115.0  # 5s safety margin before 120s kill
    
    # Each bin is represented as a list of shelves.
    # A shelf is: (z_start, d_shelf, [(x_start, w, h, z_offset_within_shelf), ...])
    # Actually, let's use a simpler representation:
    # Each bin has a list of placed boxes: [(x, y, z, w, h, d), ...]
    # And we generate candidate positions using shelf heuristics.
    
    # For efficiency, each bin maintains:
    # - list of placed boxes
    # - list of "shelf" structures for position generation
    # A shelf is at height z, has depth d_max (max d of boxes on that shelf),
    # and contains items placed along x at that shelf level.
    
    # Let's use a more structured approach:
    # bins: list of bin states
    # Each bin state: list of shelves
    # Each shelf: (z, depth_used, items) where items is list of (x, w, h)
    # When placing a box, we try each shelf in each bin, find a gap along x,
    # or create a new shelf if it fits vertically.
    
    # Actually for 3D, shelves are 2D regions. Let me think more carefully.
    
    # A cleaner model: 
    # Each bin has a set of "slots" or we just track placed boxes and 
    # generate candidate positions by checking against all placed boxes.
    
    # For 40 boxes, brute-force checking each candidate position against all
    # placed boxes in a bin is fine (40 * 40 * num_bins * num_bins ~ manageable).
    
    # Candidate position generation for a bin:
    # - Try all combinations of (x, y, z) where x, y, z are determined by
    #   the "front" of existing boxes or 0.
    # Specifically:
    #   x_candidates = {0} ∪ {x + w for each placed box}
    #   y_candidates = {0} ∪ {y + h for each placed box}
    #   z_candidates = {0} ∪ {z + d for each placed box}
    # Then for each (x, y, z) in the Cartesian product, check if the box fits
    # without overlap and within container bounds.
    
    # This can be expensive: up to (1+40)^3 = 68921 candidates per bin per box.
    # But we can prune: only consider candidates where x+w<=W, y+h<=H, z+d<=D.
    # And we can sort candidates by some heuristic (e.g., minimize x+y+z or volume wasted).
    
    # Let me implement this with some optimizations:
    # 1. Precompute candidate x, y, z values per bin
    # 2. For each candidate, do overlap check
    # 3. Pick the best (e.g., the one that minimizes the "waste" or just first valid)
    
    # Best-fit: among all valid positions across all bins, pick the one that
    # minimizes some cost. A common cost is the sum of distances from origin,
    # or the maximum coordinate used, to pack things tightly.
    
    # Let me define cost as: for a placement at (x,y,z) in a bin, 
    # cost = x * H * D + y * W * D + z * W * H  (weighted by remaining dimensions)
    # This encourages placing boxes in the "corner" closest to origin.
    
    # Actually, a simpler and effective heuristic: minimize (x, y, z) lexicographically,
    # or minimize the sum of the three coordinates.
    
    # Let me go with: for each bin, find all valid positions, pick the one with
    # minimum (x + y + z), and among bins, pick the bin with the best (minimum) such position.
    # If no bin works, create a new bin.
    
    # To make it "best-fit", we can also consider which bin the box fits "tightest" in.
    # A simple proxy: the bin where the placement has the smallest max(x+w, y+h, z+d) 
    # or smallest sum of coordinates.
    
    # Let me implement:
    # For each box (in decreasing volume order):
    #   best_pos = None
    #   best_cost = inf
    #   for each existing bin:
    #     for each candidate (x, y, z) in that bin:
    #       if fits and no overlap:
    #         cost = x + y + z  (or some other metric)
    #         if cost < best_cost:
    #           best_cost = cost
    #           best_pos = (bin_idx, x, y, z)
    #   if best_pos is None:
    #     # Create new bin, place at (0, 0, 0)
    #     best_pos = (new_bin_idx, 0, 0, 0)
    #   Place box at best_pos
    
    # For candidate generation, I'll use the "front" points approach.
    
    # Let me also add a time check: if we're running out of time, fall back to
    # a simpler heuristic for remaining boxes.
    
    # Data structures:
    # bins: list of lists of (x, y, z, w, h, d)
    # placement: dict mapping original index -> (bin_idx, x, y, z)
    
    bins = []  # list of lists of (x, y, z, w, h, d)
    placement = {}  # original index -> (bin_idx, x, y, z)
    
    def get_candidate_positions(bin_idx, w, h, d):
        """Generate candidate (x, y, z) positions for a box in a given bin."""
        if bin_idx >= len(bins):
            return []
        
        placed = bins[bin_idx]
        
        # Build candidate coordinates
        x_vals = {0}
        y_vals = {0}
        z_vals = {0}
        
        for (px, py, pz, pw, ph, pd) in placed:
            if px + pw <= W:
                x_vals.add(px + pw)
            if py + ph <= H:
                y_vals.add(py + ph)
            if pz + pd <= D:
                z_vals.add(pz + pd)
        
        # Filter to only valid starting positions
        x_vals = [x for x in x_vals if x + w <= W]
        y_vals = [y for y in y_vals if y + h <= H]
        z_vals = [z for z in z_vals if z + d <= D]
        
        if not x_vals or not y_vals or not z_vals:
            return []
        
        # Sort for deterministic behavior and to find "tight" fits first
        x_vals.sort()
        y_vals.sort()
        z_vals.sort()
        
        # Generate all combinations (with pruning)
        candidates = []
        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    # Quick bounds check
                    if x + w > W or y + h > H or z + d > D:
                        continue
                    # Check overlap with all placed boxes
                    overlap = False
                    for (px, py, pz, pw, ph, pd) in placed:
                        if not (x + w <= px or px + pw <= x or
                                y + h <= py or py + ph <= y or
                                z + d <= pz or pz + pd <= z):
                            overlap = True
                            break
                    if not overlap:
                        candidates.append((x, y, z))
        
        return candidates
    
    def check_overlap(placed, x, y, z, w, h, d):
        """Check if a box at (x,y,z) with dims (w,h,d) overlaps with any in placed."""
        for (px, py, pz, pw, ph, pd) in placed:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return True
        return False
    
    # Process boxes in decreasing volume order
    for idx, vol, w, h, d in indexed_boxes:
        # Time check
        if time.time() > deadline:
            # Fallback: place remaining boxes in a simple shelf manner
            # in the last bin or new bins
            # Just use a simple shelf for the rest
            # Find the last bin or create one
            if not bins:
                bins.append([])
                bin_idx = 0
            else:
                bin_idx = len(bins) - 1
            
            # Simple shelf placement for remaining
            x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
            # Actually, let's just try to place in existing bins with a simple check
            # Or create a new bin for each remaining box if needed
            # For safety, let's just place them in a new bin each if they don't fit
            # in the last bin, or use a very simple heuristic
            
            # Simpler fallback: create new bins as needed with shelf packing
            for idx2, vol2, w2, h2, d2 in indexed_boxes:
                if idx2 in placement:
                    continue
                # Try to place in existing bins with simple check
                placed = False
                for b_idx in range(len(bins)):
                    if check_overlap(bins[b_idx], 0, 0, 0, w2, h2, d2):
                        continue
                    # Check if it fits in the bin at origin
                    if w2 <= W and h2 <= H and d2 <= D:
                        bins[b_idx].append((0, 0, 0, w2, h2, d2))
                        placement[idx2] = (b_idx, 0, 0, 0)
                        placed = True
                        break
                if not placed:
                    # Create new bin
                    bins.append([(0, 0, 0, w2, h2, d2)])
                    placement[idx2] = (len(bins) - 1, 0, 0, 0)
            
            # Build result
            result = []
            for i in range(n):
                b_idx, x, y, z = placement[i]
                w_i, h_i, d_i = boxes[i]
                result.append((b_idx, x, y, z, w_i, h_i, d_i))
            return result
        
        # Try to place in existing bins
        best_bin = -1
        best_x, best_y, best_z = 0, 0, 0
        best_cost = float('inf')
        
        for b_idx in range(len(bins)):
            candidates = get_candidate_positions(b_idx, w, h, d)
            for (cx, cy, cz) in candidates:
                cost = cx + cy + cz
                if cost < best_cost:
                    best_cost = cost
                    best_bin = b_idx
                    best_x, best_y, best_z = cx, cy, cz
        
        if best_bin == -1:
            # Create new bin
            bins.append([])
            best_bin = len(bins) - 1
            best_x, best_y, best_z = 0, 0, 0
        
        # Place the box
        bins[best_bin].append((best_x, best_y, best_z, w, h, d))
        placement[idx] = (best_bin, best_x, best_y, best_z)
    
    # Build result in original order
    result = []
    for i in range(n):
        b_idx, x, y, z = placement[i]
        w_i, h_i, d_i = boxes[i]
        result.append((b_idx, x, y, z, w_i, h_i, d_i))
    
    return result
# EVOLVE-BLOCK-END
