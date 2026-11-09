import time
import math
import sys

_DEADLINE = 145.0
_START = None

def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _pack_eph(boxes, W, H, D, order='volume_desc'):
    """
    Pack boxes using Extreme Point Heuristics (EPH) with Best-Fit Decreasing.
    """
    indexed = list(range(len(boxes)))
    if order == 'volume_desc':
        indexed.sort(key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2])
    elif order == 'volume_asc':
        indexed.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2])
    elif order == 'largest_dim_desc':
        indexed.sort(key=lambda i: -max(boxes[i]))
    elif order == 'surface_desc':
        indexed.sort(key=lambda i: -(2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2])))
    else:
        indexed.sort(key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2])

    # bins: list of lists of (x, y, z, w, h, d)
    bins = []
    placements = [None] * len(boxes)
    
    t0 = time.time()
    deadline = _DEADLINE - 5.0 # Safety margin

    def fits(bin_boxes, x, y, z, w, h, d):
        """Check if box (w,h,d) at (x,y,z) fits in bin_bounds and doesn't overlap bin_boxes."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            # Check for overlap
            # No overlap if:
            # x+w <= bx OR bx+bw <= x
            # y+h <= by OR by+bh <= y
            # z+d <= bz OR bz+bd <= z
            if not (x + w <= bx or bx + bw <= x or
                    y + h <= by or by + bh <= y or
                    z + d <= bz or bz + bd <= z):
                return False
        return True

    def get_extreme_points(bin_boxes):
        """
        Generate non-dominated extreme points for a bin.
        A point (x,y,z) is an extreme point if it is a corner of an existing box
        (specifically the 'upper' corner relative to the origin) and is not dominated
        by another such point.
        """
        points = set()
        if not bin_boxes:
            return [(0, 0, 0)]
        
        # Candidate points are the 'top-right-back' corners of existing boxes
        # and the origin.
        points.add((0, 0, 0))
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            # The box occupies [bx, bx+bw], [by, by+bh], [bz, bz+bd]
            # Potential new positions are adjacent to these boxes.
            # Simplest robust set: The 'max' corner of each box.
            # (bx+bw, by, bz), (bx, by+bh, bz), (bx, by, bz+bd) are also candidates
            # but often dominated. The standard EPH uses the 'upper' corner (bx+bw, by+bh, bz+bd)
            # which is the point where a new box would sit on top of the existing one.
            # However, to allow filling gaps, we should consider points on the faces.
            # Let's stick to the standard: The point (bx+bw, by+bh, bz+bd) is the primary candidate.
            # But to fill side gaps, we need (bx+bw, by, bz) etc.
            # Actually, a simpler and effective heuristic is to just check the 'upper' corners
            # and the origin. If we miss side gaps, we might use more bins.
            # Let's use a slightly richer set:
            points.add((bx + bw, by, bz))
            points.add((bx, by + bh, bz))
            points.add((bx, by, bz + bd))
            points.add((bx + bw, by + bh, bz))
            points.add((bx + bw, by, bz + bd))
            points.add((bx, by + bh, bz + bd))
            points.add((bx + bw, by + bh, bz + bd))
            
        # Filter points that are within bounds
        valid_points = []
        for p in points:
            if 0 <= p[0] < W and 0 <= p[1] < H and 0 <= p[2] < D:
                valid_points.append(p)
        
        # Remove dominated points
        # P1 is dominated by P2 if P2 <= P1 component-wise.
        # We want to keep the 'minimal' points.
        # Sort by x, then y, then z.
        valid_points.sort(key=lambda p: (p[0], p[1], p[2]))
        
        non_dominated = []
        for p in valid_points:
            dominated = False
            for q in non_dominated:
                if q[0] <= p[0] and q[1] <= p[1] and q[2] <= p[2]:
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(p)
                
        return non_dominated

    for orig_idx in indexed:
        if time.time() > deadline:
            break
            
        w, h, d = boxes[orig_idx]
        
        best_bin = -1
        best_pos = None
        best_score = float('inf')
        
        for bi, bin_boxes in enumerate(bins):
            ext_points = get_extreme_points(bin_boxes)
            for (x, y, z) in ext_points:
                if fits(bin_boxes, x, y, z, w, h, d):
                    # Heuristic: Max Cube (leave largest empty cube)
                    # Or Min Waste: minimize (W-x-w)*(H-y-h)*(D-z-d) ? No, maximize it.
                    # Actually, for Best Fit, we want to minimize the 'waste' or maximize the 'fill'.
                    # Common heuristic: Maximize the remaining space in the "corner"
                    # mc = (W - x - w) * (H - y - h) * (D - z - d)
                    # We want to MINIMIZE the 'tightness' or MAXIMIZE the 'empty corner'.
                    # Let's use the "Max Cube" heuristic: maximize the volume of the largest empty axis-aligned cube.
                    # Approximation: (W - x - w) * (H - y - h) * (D - z - d) is not the largest cube.
                    # Largest cube side = min(W-x-w, H-y-h, D-z-d).
                    # Volume of that cube = min(...) ** 3.
                    # Let's use a simpler score: prefer positions closer to origin (x+y+z) and smaller 'waste'.
                    # Score = x + y + z + 0.1 * ((W-x-w)*(H-y-h)*(D-z-d)) # This is complex.
                    
                    # Standard BFD/EPH score:
                    # 1. Prefer existing bins.
                    # 2. In a bin, prefer the position that minimizes the "waste" in the corner.
                    # Waste is often approximated by the volume of the empty space "above" the box.
                    # Let's use the "Max Cube" heuristic:
                    # The largest empty axis-aligned cube that can fit in the remaining space.
                    # This is hard to compute exactly.
                    # Simpler: Minimize x + y + z (pack to bottom-left-front).
                    # Or Minimize (x + y + z) + 0.1 * (w+h+d) # No.
                    
                    # Let's use the "Waste" metric:
                    # The volume of the bounding box of the empty space is not well defined.
                    # Let's stick to: Minimize x + y + z. This tends to fill the bin from the origin.
                    # To break ties, maximize the volume of the remaining space in the "corner".
                    score = x + y + z
                    # Tie breaker: Prefer smaller 'gap' in dimensions?
                    # Let's use the product of remaining dimensions as a secondary "waste" metric to minimize.
                    # Actually, minimizing x+y+z is usually sufficient for BFD.
                    
                    if score < best_score:
                        best_score = score
                        best_bin = bi
                        best_pos = (x, y, z)
            
            # Optimization: If we found a valid position in this bin, do we need to check other bins?
            # No, we want the GLOBAL best fit across all bins.
            # However, checking all bins for all boxes is O(N * K * P).
            # With 40 boxes and small K, this is fast.
            
        if best_bin == -1:
            best_bin = len(bins)
            bins.append([])
            best_pos = (0, 0, 0)
            # New bin starts at 0,0,0. This is implicitly the best position.
        
        bins[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
        placements[orig_idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2], w, h, d)

    # Ensure all boxes are placed (in case of timeout)
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            best_bin = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            placements[i] = (best_bin, 0, 0, 0, w, h, d)

    return placements

def _validate(placements, boxes, container):
    W, H, D = container
    if len(placements) != len(boxes):
        return False
    bins = {}
    for i, (b, x, y, z, w, h, d) in enumerate(placements):
        if b is None:
            return False
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        if (w, h, d) != boxes[i]:
            return False
        bins.setdefault(b, []).append((x, y, z, w, h, d))
    for b, bin_boxes in bins.items():
        for i in range(len(bin_boxes)):
            for j in range(i + 1, len(bin_boxes)):
                x1, y1, z1, w1, h1, d1 = bin_boxes[i]
                x2, y2, z2, w2, h2, d2 = bin_boxes[j]
                if not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                        y1 + h1 <= y2 or y2 + h2 <= y1 or
                        z1 + d1 <= z2 or z2 + d2 <= z1):
                    return False
    return True

def _compute_score(placements, boxes, container):
    if not placements:
        return 0.0
    W, H, D = container
    container_vol = W * H * D
    bin_volumes = {}
    for (b, x, y, z, w, h, d) in placements:
        if b is not None:
            bin_volumes[b] = bin_volumes.get(b, 0) + w * h * d
    B = len(bin_volumes)
    if B == 0:
        return 0.0
    last_bin = max(bin_volumes.keys())
    last_vol = bin_volumes[last_bin]
    last_frac = last_vol / container_vol if container_vol > 0 else 0.0
    return B + 0.1 * (1.0 - last_frac)

def pack(boxes, container):
    W, H, D = container
    deadline = _DEADLINE - _now()
    if deadline <= 0:
        deadline = 2.0

    orders = ['volume_desc', 'largest_dim_desc', 'surface_desc', 'volume_asc']
    best_placements = None
    best_score = float('inf')

    for order in orders:
        if _now() > _DEADLINE - 5.0:
            break
        try:
            placements = _pack_eph(boxes, W, H, D, order)
            if _validate(placements, boxes, container):
                score = _compute_score(placements, boxes, container)
                if score < best_score:
                    best_score = score
                    best_placements = placements
        except Exception:
            pass

    if best_placements is None:
        # Fallback: simple first fit
        try:
            best_placements = _pack_eph(boxes, W, H, D, 'volume_desc')
        except Exception:
            best_placements = [(0, 0, 0, 0, w, h, d) for (w, h, d) in boxes]

    return best_placements
