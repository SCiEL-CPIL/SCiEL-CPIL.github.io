"""Improved 3D bin packing with Corner-Point Heuristic and deadline guard."""
import time

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 110.0  # 10s safety margin
    
    # Sort boxes by volume descending (FDD heuristic)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    
    # Bins store: (list of placed boxes, set of corner points)
    # Each placed box: (x, y, z, w, h, d)
    # Corner points are (x, y, z) tuples representing valid bottom-left-back anchors
    bins = []  # list of [boxes_list, corners_set]
    placement_map = {}  # original index -> (bin_idx, x, y, z)
    
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Fallback: place remaining boxes using corner-point heuristic quickly
            for remaining_orig_idx, (rw, rh, rd) in indexed_boxes:
                if remaining_orig_idx not in placement_map:
                    placed = False
                    for b_idx in range(len(bins)):
                        boxes_list, corners = bins[b_idx]
                        pos = _find_position_corner(corners, boxes_list, W, H, D, rw, rh, rd)
                        if pos is not None:
                            x, y, z = pos
                            placement_map[remaining_orig_idx] = (b_idx, x, y, z)
                            boxes_list.append((x, y, z, rw, rh, rd))
                            _update_corners(corners, x, y, z, rw, rh, rd, W, H, D)
                            placed = True
                            break
                    if not placed:
                        b_idx = len(bins)
                        boxes_list = [(0, 0, 0, rw, rh, rd)]
                        corners = {(0, 0, 0)}
                        # Add corners from the placed box
                        _update_corners(corners, 0, 0, 0, rw, rh, rd, W, H, D)
                        bins.append([boxes_list, corners])
                        placement_map[remaining_orig_idx] = (b_idx, 0, 0, 0)
            break
        
        placed = False
        # Try each existing bin, preferring bins with more filled space (best-fit)
        best_bin_idx = -1
        best_bin_pos = None
        best_bin_score = None
        
        for b_idx in range(len(bins)):
            boxes_list, corners = bins[b_idx]
            pos = _find_position_corner(corners, boxes_list, W, H, D, w, h, d)
            if pos is not None:
                # Score: prefer position closest to origin (compactness)
                x, y, z = pos
                pos_score = x + y + z
                if best_bin_score is None or pos_score < best_bin_score:
                    best_bin_score = pos_score
                    best_bin_idx = b_idx
                    best_bin_pos = pos
        
        if best_bin_idx >= 0:
            x, y, z = best_bin_pos
            boxes_list, corners = bins[best_bin_idx]
            placement_map[orig_idx] = (best_bin_idx, x, y, z)
            boxes_list.append((x, y, z, w, h, d))
            _update_corners(corners, x, y, z, w, h, d, W, H, D)
        else:
            # Create new bin
            b_idx = len(bins)
            boxes_list = [(0, 0, 0, w, h, d)]
            corners = {(0, 0, 0)}
            _update_corners(corners, 0, 0, 0, w, h, d, W, H, D)
            bins.append([boxes_list, corners])
            placement_map[orig_idx] = (b_idx, 0, 0, 0)
    
    # Build result in original order
    result = []
    for i in range(n):
        b_idx, x, y, z = placement_map[i]
        w, h, d = boxes[i]
        result.append((b_idx, x, y, z, w, h, d))
    
    return result

def _find_position_corner(corners, boxes_list, W, H, D, w, h, d):
    """
    Find the best corner-point position for a box using the corner-point heuristic.
    Selects the valid corner that minimizes x + y + z (closest to origin).
    """
    if not corners:
        # Empty bin: place at origin
        if w <= W and h <= H and d <= D:
            return (0, 0, 0)
        return None
    
    best_pos = None
    best_score = None
    
    for (cx, cy, cz) in corners:
        if cx + w > W or cy + h > H or cz + d > D:
            continue
        # Check if this position is valid (no overlap with existing boxes)
        if _fits(boxes_list, cx, cy, cz, w, h, d, W, H, D):
            score = cx + cy + cz
            if best_score is None or score < best_score:
                best_score = score
                best_pos = (cx, cy, cz)
    
    return best_pos

def _update_corners(corners, x, y, z, w, h, d, W, H, D):
    """
    Update the set of corner points after placing a box at (x,y,z) with dims (w,h,d).
    New corner points are generated from the outer faces of the placed box.
    A corner point is valid if it's within the container bounds.
    """
    # The new box creates potential new anchor points at its outer corners
    # For 3D packing, the relevant new "bottom-left-back" points are:
    # Points on the +x face: (x+w, y, z), (x+w, y+h, z), (x+w, y, z+d), (x+w, y+h, z+d)
    # Points on the +y face: (x, y+h, z), (x+w, y+h, z), (x, y+h, z+d), (x+w, y+h, z+d)  
    # Points on the +z face: (x, y, z+d), (x+w, y, z+d), (x, y+h, z+d), (x+w, y+h, z+d)
    
    new_corners = [
        (x + w, y, z),
        (x + w, y + h, z),
        (x + w, y, z + d),
        (x + w, y + h, z + d),
        (x, y + h, z),
        (x, y + h, z + d),
        (x, y, z + d),
        (x + w, y, z + d),
    ]
    
    for nc in new_corners:
        nx, ny, nz = nc
        if 0 <= nx <= W and 0 <= ny <= H and 0 <= nz <= D:
            corners.add(nc)

def _fits(boxes_list, x, y, z, w, h, d, W, H, D):
    """Check if a box at (x,y,z) with dims (w,h,d) fits without overlap."""
    if x + w > W or y + h > H or z + d > D:
        return False
    for (bx, by, bz, bw, bh, bd) in boxes_list:
        # Check overlap
        if not (x + w <= bx or bx + bw <= x or
                y + h <= by or by + bh <= y or
                z + d <= bz or bz + bd <= z):
            return False
    return True
# EVOLVE-BLOCK-END
