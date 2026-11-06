import time
import sys

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending for better packing
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[4])
    
    # Best placement found so far
    best_placements = None
    best_score = float('inf')
    
    def compute_score(placements):
        """Compute the score: mean B + 0.1*(1 - last-bin occupied volume fraction)"""
        if not placements:
            return 0.0
        
        # Count occupied bins
        bin_ids = set(p[0] for p in placements)
        B = len(bin_ids)
        
        # Find last bin (largest label)
        last_bin = max(bin_ids)
        
        # Compute occupied volume in last bin
        container_vol = W * H * D
        last_bin_vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == last_bin)
        
        # Score: B + 0.1*(1 - last_bin_vol / container_vol)
        # Wait, re-reading: "scored by mean B + 0.1*(1 - last-bin occupied volume fraction)"
        # B is the number of occupied bins. Mean B? Or just B?
        # "Score: mean B + 0.1 * (1 - last-bin occupied volume / container volume), where B is the number of occupied bins"
        # I think "mean B" might be a typo or it means just B. Let me re-read.
        # Actually, looking at the task description again: "scored by mean B + 0.1*(1 - last-bin occupied volume fraction)"
        # and "Score: mean B + 0.1 * (1 - last-bin occupied volume / container volume), where B is the number of occupied bins"
        # I'll interpret this as B + 0.1*(1 - last_bin_vol/container_vol)
        
        if container_vol == 0:
            return float(B)
        
        score = B + 0.1 * (1 - last_bin_vol / container_vol)
        return score
    
    def try_place_all(sorted_boxes):
        """Try to place all boxes using Best-Fit Decreasing with space partitioning."""
        # Each bin has a list of free spaces (x, y, z, w, h, d)
        bins = []  # list of lists of free spaces
        
        def new_bin():
            return [(0, 0, 0, W, H, D)]
        
        def can_place_in_space(space, w, h, d):
            sx, sy, sz, sw, sh, sd = space
            return w <= sw and h <= sh and d <= sd
        
        def split_space(space, w, h, d, px, py, pz):
            """Split a free space after placing a box at (px, py, pz) with dims (w, h, d)."""
            sx, sy, sz, sw, sh, sd = space
            
            # The box occupies [px, px+w] x [py, py+h] x [pz, pz+d]
            # within the space [sx, sx+sw] x [sy, sy+sh] x [sz, sz+sd]
            
            new_spaces = []
            
            # Split along x-axis
            if px > sx:
                # Left part
                new_spaces.append((sx, sy, sz, px - sx, sh, sd))
            if px + w < sx + sw:
                # Right part
                new_spaces.append((px + w, sy, sz, sx + sw - (px + w), sh, sd))
            
            # For the region where x overlaps with the box, split along y
            ox_start = max(px, sx)
            ox_end = min(px + w, sx + sw)
            if ox_start < ox_end:
                if py > sy:
                    # Bottom part
                    new_spaces.append((ox_start, sy, sz, ox_end - ox_start, py - sy, sd))
                if py + h < sy + sh:
                    # Top part
                    new_spaces.append((ox_start, py + h, sz, ox_end - ox_start, sy + sh - (py + h), sd))
                
                # For the region where y overlaps, split along z
                oy_start = max(py, sy)
                oy_end = min(py + h, sy + sh)
                if oy_start < oy_end:
                    if pz > sz:
                        # Back part
                        new_spaces.append((ox_start, oy_start, sz, ox_end - ox_start, oy_end - oy_start, pz - sz))
                    if pz + d < sz + sd:
                        # Front part
                        new_spaces.append((ox_start, oy_start, pz + d, ox_end - ox_start, oy_end - oy_start, sz + sd - (pz + d)))
            
            return new_spaces
        
        def find_best_position(bin_spaces, w, h, d):
            """Find the best position for a box in a bin. Returns (bin_idx, space_idx, px, py, pz) or None."""
            best = None
            best_score_val = float('inf')
            
            for si, space in enumerate(bin_spaces):
                if can_place_in_space(space, w, h, d):
                    sx, sy, sz, sw, sh, sd = space
                    # Place at the bottom-left-front of the space
                    px, py, pz = sx, sy, sz
                    # Score: prefer lower z, then lower y, then lower x (compact packing)
                    score_val = pz * 10000 + py * 100 + px
                    if score_val < best_score_val:
                        best_score_val = score_val
                        best = (si, px, py, pz)
            
            return best
        
        def place_box(bin_idx, space_idx, w, h, d, px, py, pz):
            """Place a box and update the free spaces."""
            space = bins[bin_idx][space_idx]
            new_spaces = split_space(space, w, h, d, px, py, pz)
            
            # Remove the old space and add the new ones
            bins[bin_idx].pop(space_idx)
            bins[bin_idx].extend(new_spaces)
        
        # Start with one bin
        bins.append(new_bin())
        placements = []
        
        for idx, (orig_idx, w, h, d, vol) in enumerate(sorted_boxes):
            if time.time() > deadline:
                return None  # Timeout
            
            placed = False
            
            # Try to place in existing bins
            for bi in range(len(bins)):
                result = find_best_position(bins[bi], w, h, d)
                if result is not None:
                    si, px, py, pz = result
                    place_box(bi, si, w, h, d, px, py, pz)
                    placements.append((bi, px, py, pz, w, h, d))
                    placed = True
                    break
            
            # If not placed, try a new bin
            if not placed:
                # Check if box fits in a new bin
                if w <= W and h <= H and d <= D:
                    bins.append(new_bin())
                    bi = len(bins) - 1
                    # Place at origin
                    place_box(bi, 0, w, h, d, 0, 0, 0)
                    placements.append((bi, 0, 0, 0, w, h, d))
                    placed = True
                else:
                    return None  # Box doesn't fit in any bin
            
            if not placed:
                return None
        
        return placements
    
    # Try the basic BFD approach
    result = try_place_all(indexed_boxes)
    if result is not None:
        best_placements = result
        best_score = compute_score(result)
    
    # Try a few more strategies if time permits
    # Strategy 2: Sort by max dimension descending
    if time.time() < deadline - 10:
        indexed_boxes2 = [(i, w, h, d, max(w, h, d)) for i, (w, h, d) in enumerate(boxes)]
        indexed_boxes2.sort(key=lambda x: -x[4])
        # Convert back to the format expected
        sorted_boxes2 = [(idx, w, h, d, w*h*d) for idx, w, h, d, _ in indexed_boxes2]
        result2 = try_place_all(sorted_boxes2)
        if result2 is not None:
            score2 = compute_score(result2)
            if score2 < best_score:
                best_placements = result2
                best_score = score2
    
    # Strategy 3: Sort by height descending (for shelf-like packing)
    if time.time() < deadline - 10:
        indexed_boxes3 = [(i, w, h, d, h) for i, (w, h, d) in enumerate(boxes)]
        indexed_boxes3.sort(key=lambda x: -x[4])
        sorted_boxes3 = [(idx, w, h, d, w*h*d) for idx, w, h, d, _ in indexed_boxes3]
        result3 = try_place_all(sorted_boxes3)
        if result3 is not None:
            score3 = compute_score(result3)
            if score3 < best_score:
                best_placements = result3
                best_score = score3
    
    if best_placements is None:
        # Fallback: use the original shelf heuristic
        W, H, D = container
        placements, b, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0, 0
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
    
    return best_placements
