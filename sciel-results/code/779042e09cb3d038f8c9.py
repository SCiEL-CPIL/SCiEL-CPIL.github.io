"""3D Container Loading: strong Best-Fit-Decreasing with extremal-point search."""
import time
import itertools
import heapq
import math
import random

# EVOLVE-BLOCK-START
import time as _time


def _orientations(w, h, d):
    """Yield the 6 distinct axis-permutations of (w, h, d)."""
    seen = set()
    out = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            out.append(perm)
    return out


def _overlaps(x, y, z, w, h, d, placed):
    """Return True if the box at (x,y,z) of size (w,h,d) overlaps any placed box."""
    for (px, py, pz, pw, ph, pd) in placed:
        if x < px + pw and px < x + w:
            if y < py + ph and py < y + h:
                if z < pz + pd and pz < z + d:
                    return True
    return False


def _fits_inside(x, y, z, w, h, d, W, H, D):
    return (x >= 0 and y >= 0 and z >= 0 and
            x + w <= W and y + h <= H and z + d <= D)


def _candidate_positions(placed, W, H, D, w, h, d):
    """
    Generate candidate (x,y,z) positions for a box of size (w,h,d).
    Uses extremal points: origin plus cross-product of coordinates from placed boxes,
    clipped so the box fits inside the container.
    """
    cands = set()
    # origin
    if _fits_inside(0, 0, 0, w, h, d, W, H, D):
        cands.add((0, 0, 0))
    
    # Collect unique x, y, z coordinates from all placed boxes
    xs = set([0])
    ys = set([0])
    zs = set([0])
    for (px, py, pz, pw, ph, pd) in placed:
        xs.add(px)
        xs.add(px + pw)
        ys.add(py)
        ys.add(py + ph)
        zs.add(pz)
        zs.add(pz + pd)
    
    # Clip coordinates to valid range
    xs = [x for x in xs if 0 <= x <= W - w]
    ys = [y for y in ys if 0 <= y <= H - h]
    zs = [z for z in zs if 0 <= z <= D - d]
    
    # Limit candidate set size to avoid explosion on large instances
    if len(xs) > 30:
        xs = sorted(xs)[:30]
    if len(ys) > 30:
        ys = sorted(ys)[:30]
    if len(zs) > 30:
        zs = sorted(zs)[:30]
    
    # Add all combinations
    for cx in xs:
        for cy in ys:
            for cz in zs:
                if _fits_inside(cx, cy, cz, w, h, d, W, H, D):
                    cands.add((cx, cy, cz))
    
    return cands


def _softmax(a, b, t=5.0):
    """Smooth approximation of max(a, b) via log-sum-exp."""
    m = max(a, b)
    if m == 0:
        return 0.0
    return m + math.log(1.0 + math.exp(-t * abs(a - b))) / t if a != b else m


def _find_best_position(w, h, d, W, H, D, placed):
    """
    Find the best non-overlapping position for a box of size (w,h,d).
    Uses a smooth multi-criteria score: primarily minimize z (fill from bottom),
    then a soft-compactness measure (soft-max of x+w and y+h) to encourage
    balanced footprint, with a small gap-filling bonus. Returns (x,y,z) or None.
    """
    if not placed:
        return (0, 0, 0) if _fits_inside(0, 0, 0, w, h, d, W, H, D) else None
    
    best = None
    best_score = None
    for (x, y, z) in _candidate_positions(placed, W, H, D, w, h, d):
        if _overlaps(x, y, z, w, h, d, placed):
            continue
        # Smooth surrogate for the nonsmooth max footprint:
        # soft-max encourages balanced x and y extents rather than extreme.
        ext_x = x + w
        ext_y = y + h
        footprint_smooth = _softmax(ext_x, ext_y, t=5.0)
        # Gap-filling bonus: reward positions that are close to existing boxes
        # (i.e., small distance to nearest placed box), encouraging compact clusters.
        # Use a cheap proxy: minimize sum of distances to container walls minus
        # a small bonus for being near other boxes.
        # Simpler: add a small penalty for large x+y (prefer lower-left)
        # combined with the smooth footprint.
        score = (z, footprint_smooth + 0.1 * (x + y) / max(W, H, 1), x, y)
        if best_score is None or score < best_score:
            best_score = score
            best = (x, y, z)
    return best


def _pack_bfd(boxes, W, H, D, deadline):
    """
    Best Fit Decreasing with orientation search.
    Returns (placements, placed_list) where placements is a list per original box.
    """
    n = len(boxes)
    # Sort indices by a heuristic: larger volume first, tie-break by max dimension.
    order = sorted(range(n), key=lambda i: (-(boxes[i][0]*boxes[i][1]*boxes[i][2]),
                                            -max(boxes[i])))
    placed = []          # list of (x,y,z,w,h,d)
    result = [None] * n  # per original index: (bin,x,y,z,w,h,d)

    for idx in order:
        if _time.time() >= deadline:
            break
        w, h, d = boxes[idx]
        found = False
        # Try all orientations, pick the one with the best position score
        best_orient_pos = None
        best_orient_score = None
        for (ow, oh, od) in _orientations(w, h, d):
            pos = _find_best_position(ow, oh, od, W, H, D, placed)
            if pos is not None:
                x, y, z = pos
                # Score this orientation+position: prefer low z, smooth compact footprint
                ext_x = x + ow
                ext_y = y + oh
                footprint_smooth = _softmax(ext_x, ext_y, t=5.0)
                score = (z, footprint_smooth + 0.1 * (x + y) / max(W, H, 1), x, y)
                if best_orient_score is None or score < best_orient_score:
                    best_orient_score = score
                    best_orient_pos = (x, y, z, ow, oh, od)
        if best_orient_pos is not None:
            x, y, z, ow, oh, od = best_orient_pos
            placed.append((x, y, z, ow, oh, od))
            result[idx] = (0, x, y, z, ow, oh, od)
            found = True
        if not found:
            result[idx] = (-1, 0, 0, 0, w, h, d)

    # Fill any that were never processed (deadline hit) as left out.
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    return result


def _pack_shelf(boxes, W, H, D):
    """
    A fast shelf-based fallback (similar to seed but with orientation choice).
    Returns per-box placements.
    """
    n = len(boxes)
    order = sorted(range(n), key=lambda i: (-(boxes[i][0]*boxes[i][1]*boxes[i][2]),
                                            -max(boxes[i])))
    placed = [None] * n
    x = y = z = 0
    row_h = layer_d = 0
    for idx in order:
        w, h, d = boxes[idx]
        # pick the orientation that best fits the current row/layer
        best_orient = None
        best_score = None
        for (ow, oh, od) in _orientations(w, h, d):
            # score: prefer small footprint in x, small height, small depth
            score = (x + ow > W, y + oh > H, z + od > D)
            if not any(score):
                s = (ow, oh, od)
                if best_score is None or s < best_score:
                    best_score = s
                    best_orient = (ow, oh, od)
        if best_orient is None:
            placed[idx] = (-1, 0, 0, 0, w, h, d)
            continue
        ow, oh, od = best_orient
        if x + ow > W:
            x, y, row_h = 0, y + row_h, 0
        if y + oh > H:
            y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
        if z + od > D:
            placed[idx] = (-1, 0, 0, 0, w, h, d)
            continue
        placed[idx] = (0, x, y, z, ow, oh, od)
        x, row_h, layer_d = x + ow, max(row_h, oh), max(layer_d, od)
    return placed


def _volume_filled(placements, W, H, D):
    total = 0
    for (b, x, y, z, w, h, d) in placements:
        if b == 0:
            total += w * h * d
    return total


def _validate(placements, boxes, W, H, D):
    """Check no overlaps and all inside container. Returns True if valid."""
    loaded = []
    for (b, x, y, z, w, h, d) in placements:
        if b != 0:
            continue
        if not _fits_inside(x, y, z, w, h, d, W, H, D):
            return False
        loaded.append((x, y, z, w, h, d))
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            a, b = loaded[i], loaded[j]
            if (a[0] < b[0]+b[3] and b[0] < a[0]+a[3] and
                a[1] < b[1]+b[4] and b[1] < a[1]+a[4] and
                a[2] < b[2]+b[5] and b[2] < a[2]+a[5]):
                return False
    return True


def _local_search(placements, boxes, W, H, D, deadline, max_iters=200):
    """
    Local search: try to swap or reposition boxes to improve packing.
    Returns improved placements if found.
    """
    n = len(boxes)
    current = list(placements)
    current_vol = _volume_filled(current, W, H, D)
    
    for it in range(max_iters):
        if _time.time() >= deadline:
            break
        
        # Try to find a box that could be repositioned to allow another box to fit
        # or to reduce empty space
        improved = False
        loaded_indices = [i for i in range(n) if current[i][0] == 0]
        unloaded_indices = [i for i in range(n) if current[i][0] == -1]
        
        if not unloaded_indices:
            break
        
        # Try to place one unloaded box by removing and repositioning a loaded box
        random.shuffle(unloaded_indices)
        random.shuffle(loaded_indices)
        
        for ui in unloaded_indices[:5]:
            if _time.time() >= deadline:
                break
            w, h, d = boxes[ui]
            
            # Try all orientations
            found = False
            for (ow, oh, od) in _orientations(w, h, d):
                # Try to find a position by temporarily removing some loaded boxes
                # and repositioning them
                for li in loaded_indices[:3]:
                    if _time.time() >= deadline:
                        break
                    # Save current state
                    saved_loaded = current[li]
                    saved_unloaded = current[ui]
                    
                    # Remove loaded box
                    current[li] = (-1, 0, 0, 0, boxes[li][0], boxes[li][1], boxes[li][2])
                    
                    # Build placed list without the removed box
                    placed = []
                    for i in range(n):
                        if current[i][0] == 0:
                            _, x, y, z, bw, bh, bd = current[i]
                            placed.append((x, y, z, bw, bh, bd))
                    
                    # Try to place the unloaded box
                    pos = _find_best_position(ow, oh, od, W, H, D, placed)
                    if pos is not None:
                        x, y, z = pos
                        # Check if this improves volume (it should, since we're adding a box)
                        # Try to re-place the removed box
                        pos2 = _find_best_position(boxes[li][0], boxes[li][1], boxes[li][2], W, H, D, placed + [(x, y, z, ow, oh, od)])
                        if pos2 is not None:
                            x2, y2, z2 = pos2
                            # Check for overlaps
                            new_placed = placed + [(x, y, z, ow, oh, od), (x2, y2, z2, boxes[li][0], boxes[li][1], boxes[li][2])]
                            # Validate no overlaps
                            valid = True
                            for i2 in range(len(new_placed)):
                                for j2 in range(i2+1, len(new_placed)):
                                    a, b = new_placed[i2], new_placed[j2]
                                    if (a[0] < b[0]+b[3] and b[0] < a[0]+a[3] and
                                        a[1] < b[1]+b[4] and b[1] < a[1]+a[4] and
                                        a[2] < b[2]+b[5] and b[2] < a[2]+a[5]):
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                # Apply the change
                                current[ui] = (0, x, y, z, ow, oh, od)
                                current[li] = (0, x2, y2, z2, boxes[li][0], boxes[li][1], boxes[li][2])
                                new_vol = _volume_filled(current, W, H, D)
                                if new_vol > current_vol:
                                    current_vol = new_vol
                                    improved = True
                                else:
                                    # Revert
                                    current[ui] = saved_unloaded
                                    current[li] = saved_loaded
                                found = True
                                break
                        else:
                            # Revert
                            current[li] = saved_loaded
                    else:
                        # Revert
                        current[li] = saved_loaded
                    if found:
                        break
                if found:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return current


def load(boxes, container):
    start = _time.time()
    deadline = start + 140.0  # 10s safety margin under 150s limit

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for (w, h, d) in boxes]

    # Fast shelf fallback first (cheap, always valid).
    shelf = _pack_shelf(boxes, W, H, D)
    best = shelf
    best_vol = _volume_filled(shelf, W, H, D)

    # Try the stronger BFD with extremal-point search.
    try:
        bfd = _pack_bfd(boxes, W, H, D, deadline)
        if _validate(bfd, boxes, W, H, D):
            vol = _volume_filled(bfd, W, H, D)
            if vol > best_vol:
                best = bfd
                best_vol = vol
    except Exception:
        pass

    # If we still have time, try a second BFD pass with a different sort
    # (by max-dimension) to diversify, keeping the better result.
    if _time.time() < deadline - 5.0:
        try:
            n = len(boxes)
            order2 = sorted(range(n), key=lambda i: (-max(boxes[i]),
                                                     -(boxes[i][0]*boxes[i][1]*boxes[i][2])))
            placed = []
            result = [None] * n
            for idx in order2:
                if _time.time() >= deadline - 3.0:
                    break
                w, h, d = boxes[idx]
                found = False
                best_orient_pos = None
                best_orient_score = None
                for (ow, oh, od) in _orientations(w, h, d):
                    pos = _find_best_position(ow, oh, od, W, H, D, placed)
                    if pos is not None:
                        x, y, z = pos
                        ext_x = x + ow
                        ext_y = y + oh
                        footprint_smooth = _softmax(ext_x, ext_y, t=5.0)
                        score = (z, footprint_smooth + 0.1 * (x + y) / max(W, H, 1), x, y)
                        if best_orient_score is None or score < best_orient_score:
                            best_orient_score = score
                            best_orient_pos = (x, y, z, ow, oh, od)
                if best_orient_pos is not None:
                    x, y, z, ow, oh, od = best_orient_pos
                    placed.append((x, y, z, ow, oh, od))
                    result[idx] = (0, x, y, z, ow, oh, od)
                    found = True
                if not found:
                    result[idx] = (-1, 0, 0, 0, w, h, d)
            for i in range(n):
                if result[i] is None:
                    w, h, d = boxes[i]
                    result[i] = (-1, 0, 0, 0, w, h, d)
            if _validate(result, boxes, W, H, D):
                vol = _volume_filled(result, boxes, W, H, D)
                if vol > best_vol:
                    best = result
                    best_vol = vol
        except Exception:
            pass

    # Try local search to improve the best solution found so far
    if _time.time() < deadline - 10.0:
        try:
            improved = _local_search(best, boxes, W, H, D, deadline - 10.0, max_iters=100)
            if _validate(improved, boxes, W, H, D):
                vol = _volume_filled(improved, boxes, W, H, D)
                if vol > best_vol:
                    best = improved
                    best_vol = vol
        except Exception:
            pass

    # Final safety: ensure validity; if invalid, fall back to shelf.
    if not _validate(best, boxes, W, H, D):
        best = shelf

    return best
# EVOLVE-BLOCK-END
