import time
import random
import numpy as np
from collections import defaultdict

DEADLINE = time.time() + 145.0


def _check_overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
    """Check if two boxes overlap."""
    return (x1 < x2 + w2 and x1 + w1 > x2 and
            y1 < y2 + h2 and y1 + h1 > y2 and
            z1 < z2 + d2 and z1 + d1 > z2)


def _fits_in_container(x, y, z, w, h, d, W, H, D):
    """Check if box fits in container at given position."""
    return (x >= 0 and y >= 0 and z >= 0 and
            x + w <= W and y + h <= H and z + d <= D)


def _can_place(placement, boxes, container, bin_id, pos_x, pos_y, pos_z):
    """Check if a box can be placed at the given position in the given bin."""
    W, H, D = container
    idx, w, h, d = placement
    if not _fits_in_container(pos_x, pos_y, pos_z, w, h, d, W, H, D):
        return False
    for (px, py, pz, pw, ph, pd) in boxes[bin_id]:
        if _check_overlap(pos_x, pos_y, pos_z, w, h, d, px, py, pz, pw, ph, pd):
            return False
    return True


def _get_anchors(boxes_in_bin, W, H, D):
    """Generate candidate anchor positions for placing a new box."""
    if not boxes_in_bin:
        return [(0, 0, 0)]
    
    anchors = [(0, 0, 0)]
    for (px, py, pz, pw, ph, pd) in boxes_in_bin:
        # Faces
        if px + pw < W:
            anchors.append((px + pw, py, pz))
        if py + ph < H:
            anchors.append((px, py + ph, pz))
        if pz + pd < D:
            anchors.append((px, py, pz + pd))
        
        # Edges (combinations of two faces)
        if px + pw < W and py + ph < H:
            anchors.append((px + pw, py + ph, pz))
        if px + pw < W and pz + pd < D:
            anchors.append((px + pw, py, pz + pd))
        if py + ph < H and pz + pd < D:
            anchors.append((px, py + ph, pz + pd))
        
        # Corner (all three faces)
        if px + pw < W and py + ph < H and pz + pd < D:
            anchors.append((px + pw, py + ph, pz + pd))
    
    # Deduplicate
    unique_anchors = list(set(anchors))
    return unique_anchors


def _score_position(pos, w, h, d, boxes_in_bin, W, H, D, mode='best_fit'):
    """Score a candidate position for placing a box."""
    x, y, z = pos
    new_max_x = x + w
    new_max_y = y + h
    new_max_z = z + d
    
    for (px, py, pz, pw, ph, pd) in boxes_in_bin:
        new_max_x = max(new_max_x, px + pw)
        new_max_y = max(new_max_y, py + ph)
        new_max_z = max(new_max_z, pz + pd)
    
    # Bounding box volume
    bbox_vol = new_max_x * new_max_y * new_max_z
    
    if mode == 'best_fit':
        # Minimize bounding box volume, then prefer lower z, then smaller x, y
        score = bbox_vol
        score += z * 1.0
        score += x * 0.1
        score += y * 0.01
        return score
    elif mode == 'surface_area':
        # Minimize surface area of bounding box
        sa = 2 * (new_max_x * new_max_y + new_max_x * new_max_z + new_max_y * new_max_z)
        return sa
    elif mode == 'contact':
        # Maximize contact with existing boxes (minimize gap)
        min_gap = float('inf')
        for (px, py, pz, pw, ph, pd) in boxes_in_bin:
            # Check if boxes are adjacent
            gap_x = max(0, max(x - (px + pw), px - (x + w)))
            gap_y = max(0, max(y - (py + ph), py - (y + h)))
            gap_z = max(0, max(z - (pz + pd), pz - (z + d)))
            if gap_x == 0 and gap_y == 0 and gap_z == 0:
                min_gap = 0
            else:
                # Distance to nearest face
                dist = min(gap_x, gap_y, gap_z)
                if dist < min_gap:
                    min_gap = dist
        if min_gap == float('inf'):
            min_gap = 0
        return min_gap
    else:
        return bbox_vol


def _pack_single_order(boxes, container, order, max_time=20.0):
    """Pack boxes in a given order using best-fit placement."""
    W, H, D = container
    n = len(boxes)
    t_start = time.time()
    
    bins = []  # List of lists of (x, y, z, w, h, d)
    box_to_bin = {}  # box index -> (bin_id, pos_in_bin)
    
    modes = ['best_fit', 'surface_area', 'contact']
    
    for idx in order:
        if time.time() > t_start + max_time:
            # Fallback: place remaining boxes one per bin
            while idx in [i for i in range(n) if i not in box_to_bin]:
                pass
            break
        
        w, h, d = boxes[idx]
        
        # Try to place in existing bins first
        placed = False
        for mode in modes:
            if placed:
                break
            best_score = float('inf')
            best_pos = None
            best_bin = -1
            
            for bi, bin_boxes in enumerate(bins):
                anchors = _get_anchors(bin_boxes, W, H, D)
                for (ax, ay, az) in anchors:
                    if not _fits_in_container(ax, ay, az, w, h, d, W, H, D):
                        continue
                    # Check overlap
                    overlaps = False
                    for (px, py, pz, pw, ph, pd) in bin_boxes:
                        if _check_overlap(ax, ay, az, w, h, d, px, py, pz, pw, ph, pd):
                            overlaps = True
                            break
                    if overlaps:
                        continue
                    
                    score = _score_position((ax, ay, az), w, h, d, bin_boxes, W, H, D, mode)
                    if score < best_score:
                        best_score = score
                        best_pos = (ax, ay, az)
                        best_bin = bi
            
            if best_pos is not None:
                x, y, z = best_pos
                bins[best_bin].append((x, y, z, w, h, d))
                box_to_bin[idx] = (best_bin, len(bins[best_bin]) - 1)
                placed = True
        
        if not placed:
            # Start a new bin
            bins.append([(0, 0, 0, w, h, d)])
            box_to_bin[idx] = (len(bins) - 1, 0)
    
    # Handle any boxes not placed due to timeout
    for i in range(n):
        if i not in box_to_bin:
            bins.append([(0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])])
            box_to_bin[i] = (len(bins) - 1, 0)
    
    return bins, box_to_bin


def _score_solution(bins, container):
    """Calculate the official score: B + 0.1 * (1 - last_bin_fill)."""
    W, H, D = container
    C = W * H * D
    B = len(bins)
    
    if B == 0:
        return 0.0
        
    last_bin = bins[-1]
    if not last_bin:
        return B
        
    last_vol = sum(w * h * d for (x, y, z, w, h, d) in last_bin)
    fill_ratio = last_vol / C if C > 0 else 0.0
    
    return B + 0.1 * (1.0 - fill_ratio)


def _local_search(bins, boxes, container, max_iters=500, rng=None):
    """Improve packing by moving boxes between bins or repositioning."""
    W, H, D = container
    if rng is None:
        rng = random.Random(42)
    
    n = len(boxes)
    # Rebuild box_to_bin mapping
    box_to_bin = {}
    for bi, bin_boxes in enumerate(bins):
        for pi, (x, y, z, w, h, d) in enumerate(bin_boxes):
            # Find which box this is (by dimensions and position)
            # This is tricky - we need to track indices
            pass
    
    # For simplicity, let's try a different local search approach:
    # Try moving a box from a later bin to an earlier bin if it fits
    current_bins = [list(bin_boxes) for bin_boxes in bins]
    current_score = _score_solution(current_bins, container)
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        # Pick a random box from a later bin
        if len(current_bins) < 2:
            break
        
        source_bin = rng.randint(1, len(current_boxes) - 1)
        if source_bin >= len(current_bins):
            continue
        if not current_bins[source_bin]:
            continue
        
        box_idx_in_bin = rng.randint(0, len(current_bins[source_bin]) - 1)
        x, y, z, w, h, d = current_bins[source_bin][box_idx_in_bin]
        
        # Try to remove this box and place it in an earlier bin
        # Temporarily remove
        current_bins[source_bin].pop(box_idx_in_bin)
        
        # Try to place in each earlier bin
        found = False
        for target_bin in range(source_bin):
            anchors = _get_anchors(current_bins[target_bin], W, H, D)
            for (ax, ay, az) in anchors:
                if not _fits_in_container(ax, ay, az, w, h, d, W, H, D):
                    continue
                overlaps = False
                for (px, py, pz, pw, ph, pd) in current_bins[target_bin]:
                    if _check_overlap(ax, ay, az, w, h, d, px, py, pz, pw, ph, pd):
                        overlaps = True
                        break
                if overlaps:
                    continue
                
                # Try placing here
                current_bins[target_bin].append((ax, ay, az, w, h, d))
                new_score = _score_solution(current_bins, container)
                if new_score < current_score - 1e-9:
                    current_score = new_score
                    found = True
                    break
                else:
                    current_bins[target_bin].pop()
            
            if found:
                break
        
        if not found:
            # Put the box back
            current_bins[source_bin].insert(box_idx_in_bin, (x, y, z, w, h, d))
        
        # Remove empty bins
        current_bins = [b for b in current_bins if b]
    
    return current_bins


def _generate_orderings(boxes, n, rng):
    """Generate diverse orderings of box indices."""
    idx = list(range(n))
    
    # 1. Volume Descending
    yield sorted(idx, key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    # 2. Max Dimension Descending
    yield sorted(idx, key=lambda i: max(boxes[i]), reverse=True)
    
    # 3. Sum of Dimensions Descending
    yield sorted(idx, key=lambda i: sum(boxes[i]), reverse=True)
    
    # 4. Surface Area Descending
    yield sorted(idx, key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][0]*boxes[i][2] + boxes[i][1]*boxes[i][2]), reverse=True)
    
    # 5. Aspect Ratio Descending
    yield sorted(idx, key=lambda i: max(boxes[i]) / max(min(boxes[i]), 1), reverse=True)
    
    # 6-15. Random Shuffles with different seeds
    for _ in range(10):
        s = idx[:]
        rng.shuffle(s)
        yield s
    
    # 16. Weighted random (larger boxes more likely early)
    weights = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    s = idx[:]
    for i in range(n - 1, 0, -1):
        # Weighted selection
        r = rng.random() * sum(weights[:i+1])
        cumsum = 0
        for j in range(i + 1):
            cumsum += weights[j]
            if r <= cumsum:
                s[i], s[j] = s[j], s[i]
                break
    yield s


def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    best_bins = None
    best_score = float('inf')
    
    t_start = time.time()
    rng = random.Random(42)
    
    # Generate and try multiple orderings
    for order in _generate_orderings(boxes, n, rng):
        if time.time() > DEADLINE - 10:
            break
        
        try:
            bins, _ = _pack_single_order(boxes, container, order, max_time=15.0)
            score = _score_solution(bins, container)
            
            if score < best_score:
                best_score = score
                best_bins = [list(b) for b in bins]
                
                # Try local search if time permits
                if time.time() < DEADLINE - 20:
                    improved_bins = _local_search(best_bins, boxes, container, max_iters=200, rng=rng)
                    improved_score = _score_solution(improved_bins, container)
                    if improved_score < best_score - 1e-9:
                        best_score = improved_score
                        best_bins = improved_bins
                        
        except Exception:
            continue
    
    # Fallback if no valid solution found
    if best_bins is None:
        best_bins = [[(0, 0, 0, w, h, d)] for w, h, d in boxes]
        best_score = _score_solution(best_bins, container)
    
    # Construct final placement list
    # We need to map each box to its (bin, x, y, z, w, h, d)
    # Since we don't track original indices in bins, we need to reconstruct
    
    # For each box i, find which bin and position it's in
    # This is ambiguous if boxes have same dimensions, so let's use a different approach
    # We'll assume the order in bins corresponds to the packing order
    
    # Actually, let's re-run the best packing with index tracking
    # For simplicity, let's just assign boxes to bins in order
    result = [None] * n
    bin_counter = 0
    for bi, bin_boxes in enumerate(best_bins):
        for pi, (x, y, z, w, h, d) in enumerate(bin_boxes):
            # Find the original index of this box
            # This is tricky - let's use a heuristic: match by dimensions
            # For now, just assign sequentially
            pass
    
    # Better approach: re-run the best packing and track indices
    # Let's do a simple reconstruction: assume boxes are placed in order
    # and match by dimensions (first match wins)
    
    used_boxes = set()
    for i in range(n):
        w, h, d = boxes[i]
        found = False
        for bi, bin_boxes in enumerate(best_bins):
            for pi, (x, y, z, bw, bh, bd) in enumerate(bin_boxes):
                if bw == w and bh == h and bd == d and (bi, pi) not in used_boxes:
                    result[i] = (bi, x, y, z, w, h, d)
                    used_boxes.add((bi, pi))
                    found = True
                    break
            if found:
                break
        if not found:
            # Should not happen, but fallback
            result[i] = (0, 0, 0, w, h, d)
    
    return result
