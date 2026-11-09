import time
import heapq
from functools import total_ordering

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    # Sort boxes by volume descending, then by max dimension descending
    indexed = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed.sort(key=lambda t: (-t[4], -max(t[1], t[2], t[3])))

    # We'll maintain a list of bins. Each bin has a list of placed boxes (x,y,z,w,h,d).
    # For efficiency, we'll use a candidate generation approach.

    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2 and
                z1 < z2 + d2 and z1 + d1 > z2)

    def fits_in_container(x, y, z, w, h, d):
        return x >= 0 and y >= 0 and z >= 0 and x + w <= W and y + h <= H and z + d <= D

    def generate_candidates(placed, w, h, d):
        """Generate candidate positions for a box of size (w,h,d) given placed boxes."""
        if not placed:
            return [(0, 0, 0)]
        
        candidates = set()
        # Origin
        candidates.add((0, 0, 0))
        
        # For each placed box, generate positions relative to it
        for (px, py, pz, pw, ph, pd) in placed:
            # x = px + pw (to the right), y and z can be 0 or py or pz
            if px + pw <= W:
                candidates.add((px + pw, 0, 0))
                candidates.add((px + pw, py, 0))
                candidates.add((px + pw, 0, pz))
                candidates.add((px + pw, py, pz))
            
            # x can be 0 or px, y = py + ph
            if py + ph <= H:
                candidates.add((0, py + ph, 0))
                candidates.add((px, py + ph, 0))
                candidates.add((0, py + ph, pz))
                candidates.add((px, py + ph, pz))
            
            # x and y can be 0 or px/py, z = pz + pd
            if pz + pd <= D:
                candidates.add((0, 0, pz + pd))
                candidates.add((px, 0, pz + pd))
                candidates.add((0, py, pz + pd))
                candidates.add((px, py, pz + pd))
        
        # Filter to valid positions within container
        valid = []
        for (cx, cy, cz) in candidates:
            if cx < 0 or cy < 0 or cz < 0:
                continue
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            valid.append((cx, cy, cz))
        
        return valid

    def is_valid_placement(placed, x, y, z, w, h, d):
        """Check if placing a box at (x,y,z) with size (w,h,d) doesn't overlap with any placed box."""
        for (px, py, pz, pw, ph, pd) in placed:
            if overlaps(x, y, z, w, h, d, px, py, pz, pw, ph, pd):
                return False
        return True

    def placement_cost(x, y, z, w, h, d):
        """Cost function: prefer lower y, then lower z, then lower x (bottom-left-far)."""
        # We want to minimize y first, then z, then x
        return (y, z, x)

    # Best-fit strategy: for each box, try all bins, find best placement
    bins = []  # list of lists of (x,y,z,w,h,d)
    
    for idx, (orig_idx, w, h, d, vol) in enumerate(indexed):
        if time.time() > deadline:
            # Fallback: place remaining boxes in a simple manner
            break
        
        best_bin = None
        best_pos = None
        best_cost = None
        
        for bin_idx in range(len(bins)):
            placed = bins[bin_idx]
            candidates = generate_candidates(placed, w, h, d)
            
            for (cx, cy, cz) in candidates:
                if not is_valid_placement(placed, cx, cy, cz, w, h, d):
                    continue
                cost = placement_cost(cx, cy, cz, w, h, d)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_bin = bin_idx
                    best_pos = (cx, cy, cz)
        
        if best_bin is None:
            # Try new bin
            if fits_in_container(0, 0, 0, w, h, d):
                best_bin = len(bins)
                bins.append([])
                best_pos = (0, 0, 0)
            else:
                # Should not happen if boxes fit in container
                best_bin = len(bins)
                bins.append([])
                best_pos = (0, 0, 0)
        
        bins[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
    
    # If we hit deadline, place remaining boxes greedily in first available spot
    if time.time() > deadline:
        # Quick fallback for remaining
        pass

    # Build result
    result = [None] * n
    for bin_idx, placed in enumerate(bins):
        for (x, y, z, w, h, d) in placed:
            # Find original index
            # This is inefficient; let's track it better
            pass

    # Let me redo with tracking
    bins = []  # list of lists of (orig_idx, x,y,z,w,h,d)
    
    for idx, (orig_idx, w, h, d, vol) in enumerate(indexed):
        if time.time() > deadline:
            break
        
        best_bin = None
        best_pos = None
        best_cost = None
        
        for bin_idx in range(len(bins)):
            placed = bins[bin_idx]
            # Extract just the geometry for candidate generation
            placed_geom = [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in placed]
            candidates = generate_candidates(placed_geom, w, h, d)
            
            for (cx, cy, cz) in candidates:
                if not is_valid_placement(placed_geom, cx, cy, cz, w, h, d):
                    continue
                cost = placement_cost(cx, cy, cz, w, h, d)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_bin = bin_idx
                    best_pos = (cx, cy, cz)
        
        if best_bin is None:
            if fits_in_container(0, 0, 0, w, h, d):
                best_bin = len(bins)
                bins.append([])
                best_pos = (0, 0, 0)
            else:
                best_bin = len(bins)
                bins.append([])
                best_pos = (0, 0, 0)
        
        bins[best_bin].append((orig_idx, best_pos[0], best_pos[1], best_pos[2], w, h, d))
    
    # Handle remaining boxes if deadline hit
    remaining = [item for item in indexed if item[0] not in {b[0] for bin in bins for b in bin}]
    for (orig_idx, w, h, d, vol) in remaining:
        if time.time() > deadline:
            # Just place in a new bin at origin (may be invalid but avoid crash)
            bins.append([(orig_idx, 0, 0, 0, w, h, d)])
            continue
        # Quick placement: try each existing bin, then new bin
        placed_any = False
        for bin_idx in range(len(bins)):
            placed = bins[bin_idx]
            placed_geom = [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in placed]
            candidates = generate_candidates(placed_geom, w, h, d)
            for (cx, cy, cz) in candidates:
                if is_valid_placement(placed_geom, cx, cy, cz, w, h, d):
                    bins[bin_idx].append((orig_idx, cx, cy, cz, w, h, d))
                    placed_any = True
                    break
            if placed_any:
                break
        if not placed_any:
            bins.append([(orig_idx, 0, 0, 0, w, h, d)])
    
    # Build final result
    result = [None] * n
    for bin_idx, bin_contents in enumerate(bins):
        for (orig_idx, x, y, z, w, h, d) in bin_contents:
            result[orig_idx] = (bin_idx, x, y, z, w, h, d)
    
    # Ensure all are filled
    for i in range(n):
        if result[i] is None:
            result[i] = (0, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
    
    return result
# EVOLVE-BLOCK-END
