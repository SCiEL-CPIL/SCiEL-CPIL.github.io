"""Improved 3D bin packing using best-fit decreasing with corner candidate search."""
import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def _box_volume(w, h, d):
    return w * h * d

def _overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                y1 + h1 <= y2 or y2 + h2 <= y1 or
                z1 + d1 <= z2 or z2 + d2 <= z1)

def _fits(x, y, z, w, h, d, W, H, D):
    return x >= 0 and y >= 0 and z >= 0 and x + w <= W and y + h <= H and z + d <= D

def _corner_candidates(placed, W, H, D):
    """Generate candidate corner positions from existing placements."""
    candidates = set()
    candidates.add((0, 0, 0))
    for (bx, by, bz, bw, bh, bd) in placed:
        # Right face
        candidates.add((bx + bw, by, bz))
        # Front face
        candidates.add((bx, by + bh, bz))
        # Top face
        candidates.add((bx, by, bz + bd))
        # Edges
        candidates.add((bx + bw, by + bh, bz))
        candidates.add((bx + bw, by, bz + bd))
        candidates.add((bx, by + bh, bz + bd))
        candidates.add((bx + bw, by + bh, bz + bd))
    
    # Filter to valid positions within container
    valid = []
    for (cx, cy, cz) in candidates:
        if 0 <= cx < W and 0 <= cy < H and 0 <= cz < D:
            valid.append((cx, cy, cz))
    return valid

def _find_best_position(placed, w, h, d, W, H, D):
    """Find the best (lowest, then leftmost, then frontmost) position for a box."""
    candidates = _corner_candidates(placed, W, H, D)
    
    best_pos = None
    best_score = None
    
    for (cx, cy, cz) in candidates:
        if not _fits(cx, cy, cz, w, h, d, W, H, D):
            continue
        # Check overlap with all placed boxes
        overlap = False
        for (px, py, pz, pw, ph, pd) in placed:
            if _overlaps(cx, cy, cz, w, h, d, px, py, pz, pw, ph, pd):
                overlap = True
                break
        if overlap:
            continue
        # Score: prefer lower z, then lower y, then lower x (minimize height usage)
        score = (cz, cy, cx)
        if best_score is None or score < best_score:
            best_score = score
            best_pos = (cx, cy, cz)
    
    return best_pos

def pack(boxes, container):
    """
    Place boxes into bins using best-fit decreasing heuristic with corner candidate search.
    Returns list of (bin_id, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Sort boxes by volume descending (best-fit decreasing)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda ib: _box_volume(ib[1][0], ib[1][1], ib[1][2]), reverse=True)
    
    # Each bin: list of (x, y, z, w, h, d)
    bins = []
    bin_volumes = []
    
    placements = [None] * len(boxes)
    
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            break
        
        placed_in_bin = False
        best_bin = -1
        best_pos = None
        best_bin_score = None
        
        # Try each existing bin
        for bi, placed in enumerate(bins):
            if time.time() > deadline:
                break
            
            pos = _find_best_position(placed, w, h, d, W, H, D)
            if pos is not None:
                # Score: prefer bin with more volume already used (tighter packing)
                # and position with lower z, y, x
                bin_vol = bin_volumes[bi]
                pos_score = (pos[2], pos[1], pos[0])
                # We want to minimize: (negative bin_vol to prefer full bins, pos_score)
                # Actually, prefer bins that are more full to reduce bin count
                combined = (-bin_vol, pos_score)
                if best_bin_score is None or combined < best_bin_score:
                    best_bin_score = combined
                    best_bin = bi
                    best_pos = pos
        
        # If no existing bin works, try a new bin
        if best_bin == -1:
            # Check if box fits in an empty bin
            if w <= W and h <= H and d <= D:
                best_bin = len(bins)
                best_pos = (0, 0, 0)
        
        if best_pos is not None:
            if best_bin == len(bins):
                bins.append([])
                bin_volumes.append(0)
            
            bx, by, bz = best_pos
            bins[best_bin].append((bx, by, bz, w, h, d))
            bin_volumes[best_bin] += w * h * d
            placements[orig_idx] = (best_bin, bx, by, bz, w, h, d)
        else:
            # Box doesn't fit anywhere - should not happen with valid input
            # Place it in a new bin at origin as fallback
            new_bin = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            bin_volumes.append(w * h * d)
            placements[orig_idx] = (new_bin, 0, 0, 0, w, h, d)
    
    # Fill any unplaced boxes (shouldn't happen)
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            new_bin = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            bin_volumes.append(w * h * d)
            placements[i] = (new_bin, 0, 0, 0, w, h, d)
    
    return placements

# EVOLVE-BLOCK-END
