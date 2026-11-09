"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import heapq
from itertools import product

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    W, H, D = container
    n = len(boxes)
    
    # Sort boxes by volume descending (best-fit decreasing)
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    # Each bin stores a list of placed boxes: (x, y, z, w, h, d)
    bins = []
    
    def boxes_overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return (x1 < x2 + w2 and x2 < x1 + w1 and
                y1 < y2 + h2 and y2 < y1 + h1 and
                z1 < z2 + d2 and z2 < z1 + d1)
    
    def find_best_position(bin_boxes, w, h, d, W, H, D):
        """Find the best valid (x, y, z) position in the bin using bottom-left-far heuristic.
        Returns (x, y, z) or None if no valid position exists."""
        if w > W or h > H or d > D:
            return None
        
        # Generate candidate positions from all combinations of x, y, z boundaries
        x_candidates = {0}
        y_candidates = {0}
        z_candidates = {0}
        
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            x_candidates.add(bx)
            x_candidates.add(bx + bw)
            y_candidates.add(by)
            y_candidates.add(by + bh)
            z_candidates.add(bz)
            z_candidates.add(bz + bd)
        
        # Filter candidates to valid ranges
        x_candidates = {x for x in x_candidates if 0 <= x <= W - w}
        y_candidates = {y for y in y_candidates if 0 <= y <= H - h}
        z_candidates = {z for z in z_candidates if 0 <= z <= D - d}
        
        if not x_candidates or not y_candidates or not z_candidates:
            return None
        
        best_pos = None
        best_score = float('inf')
        
        # Try all combinations of candidate positions
        for x in sorted(x_candidates):
            for y in sorted(y_candidates):
                for z in sorted(z_candidates):
                    # Check if this position is valid (no overlap)
                    valid = True
                    for (bx, by, bz, bw, bh, bd) in bin_boxes:
                        if boxes_overlap(x, y, z, w, h, d, bx, by, bz, bw, bh, bd):
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Bottom-left-far scoring: prefer positions with smaller z, then y, then x
                    # This packs boxes toward the bottom, left, and far (back)
                    score = z * 1000000 + y * 1000 + x
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
        
        return best_pos
    
    # Place each box
    placements = [None] * n  # (bin_idx, x, y, z, w, h, d)
    
    for orig_idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break
        
        best_bin = -1
        best_pos = None
        best_bin_score = float('inf')
        
        # Try existing bins
        for bi, bin_boxes in enumerate(bins):
            pos = find_best_position(bin_boxes, w, h, d, W, H, D)
            if pos is not None:
                # Score: prefer bins that are already more filled (less empty space)
                bin_vol = sum(bw*bh*bd for (_, _, _, bw, bh, bd) in bin_boxes)
                # Prefer bins with higher existing volume (more compact packing)
                bin_score = -bin_vol
                if bin_score < best_bin_score:
                    best_bin_score = bin_score
                    best_bin = bi
                    best_pos = pos
        
        if best_bin == -1:
            # Open a new bin
            best_bin = len(bins)
            bins.append([])
            best_pos = (0, 0, 0)
        
        # Place the box
        x, y, z = best_pos
        bins[best_bin].append((x, y, z, w, h, d))
        placements[orig_idx] = (best_bin, x, y, z, w, h, d)
    
    # Local optimization: try to move boxes to better positions
    if time.time() < deadline - 10:
        for _ in range(3):  # Multiple passes
            if time.time() > deadline - 5:
                break
            improved = False
            for bi, bin_boxes in enumerate(bins):
                if not bin_boxes:
                    continue
                # Try to improve each box's position
                for idx in range(len(bin_boxes)):
                    if time.time() > deadline - 5:
                        break
                    bx, by, bz, bw, bh, bd = bin_boxes[idx]
                    orig_idx = None
                    # Find original index for this box
                    for oi, (ow, oh, od) in enumerate(boxes):
                        if placements[oi] is not None and placements[oi][0] == bi:
                            px, py, pz = placements[oi][1], placements[oi][2], placements[oi][3]
                            if px == bx and py == by and pz == bz:
                                orig_idx = oi
                                break
                    if orig_idx is None:
                        continue
                    
                    # Remove box temporarily
                    bin_boxes.pop(idx)
                    new_pos = find_best_position(bin_boxes, bw, bh, bd, W, H, D)
                    if new_pos is not None:
                        nx, ny, nz = new_pos
                        # Check if new position is better (lower z, y, x)
                        if (nz < bz) or (nz == bz and ny < by) or (nz == bz and ny == by and nx < bx):
                            bin_boxes.append((nx, ny, nz, bw, bh, bd))
                            placements[orig_idx] = (bi, nx, ny, nz, bw, bh, bd)
                            improved = True
                        else:
                            bin_boxes.append((bx, by, bz, bw, bh, bd))
                    else:
                        bin_boxes.append((bx, by, bz, bw, bh, bd))
                    if not improved and time.time() > deadline - 5:
                        break
            if not improved:
                break
    
    # Handle any unplaced boxes
    if any(p is None for p in placements):
        remaining = [i for i in range(n) if placements[i] is None]
        for i in remaining:
            w, h, d = boxes[i]
            placed = False
            for bi, bin_boxes in enumerate(bins):
                pos = find_best_position(bin_boxes, w, h, d, W, H, D)
                if pos is not None:
                    x, y, z = pos
                    bins[bi].append((x, y, z, w, h, d))
                    placements[i] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
            if not placed:
                bi = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placements[i] = (bi, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
