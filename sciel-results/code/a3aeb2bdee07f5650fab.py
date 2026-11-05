"""Rewritten program for 3D Container Loading with stronger packing heuristics."""
import time
import math
import random
import heapq

# EVOLVE-BLOCK-START

def _all_orientations(w, h, d):
    """Return all 6 unique orientations of a box (w,h,d)."""
    seen = set()
    result = []
    for o in [(w,h,d),(w,d,h),(h,w,d),(h,d,w),(d,w,h),(d,h,w)]:
        if o not in seen:
            seen.add(o)
            result.append(o)
    return result

def _fits_at(x, y, z, w, h, d, W, H, D, placed):
    """Check if box (x,y,z,w,h,d) fits in container without overlapping placed boxes."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if not (x + w <= px or px + pw <= x or
                y + h <= py or py + ph <= y or
                z + d <= pz or pz + pd <= z):
            return False
    return True

def _find_best_position(w, h, d, W, H, D, placed, strategy='best_fit'):
    """
    Find the best position for a box of size (w,h,d) in the container.
    Returns (x,y,z) or None.
    
    Strategies:
    - 'best_fit': minimize the maximum extent (x+w, y+h, z+d) to keep bounding box tight
    - 'first_fit': first position in scan order (z, y, x)
    - 'min_gap': minimize total gap around the box
    """
    best_pos = None
    best_score = None
    
    # Generate candidate positions efficiently
    # Instead of scanning all positions, we can be smarter
    # But for correctness and simplicity, let's scan but with early termination
    
    max_z = D - d + 1
    max_y = H - h + 1
    max_x = W - w + 1
    
    if max_z <= 0 or max_y <= 0 or max_x <= 0:
        return None
    
    if strategy == 'first_fit':
        # First fit: scan z, then y, then x
        for z in range(max_z):
            for y in range(max_y):
                for x in range(max_x):
                    if _fits_at(x, y, z, w, h, d, W, H, D, placed):
                        return (x, y, z)
        return None
    elif strategy == 'best_fit':
        # Best fit: minimize max(x+w, y+h, z+d)
        for z in range(max_z):
            for y in range(max_y):
                for x in range(max_x):
                    if _fits_at(x, y, z, w, h, d, W, H, D, placed):
                        score = max(x + w, y + h, z + d)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        return best_pos
    elif strategy == 'min_gap':
        # Minimize total gap: (W-(x+w)) + (H-(y+h)) + (D-(z+d))
        for z in range(max_z):
            for y in range(max_y):
                for x in range(max_x):
                    if _fits_at(x, y, z, w, h, d, W, H, D, placed):
                        gap = (W - (x + w)) + (H - (y + h)) + (D - (z + d))
                        if best_score is None or gap < best_score:
                            best_score = gap
                            best_pos = (x, y, z)
        return best_pos
    
    return None

def _pack_first_fit_decreasing(boxes, W, H, D, sort_key=None, placement_strategy='best_fit', deadline=None):
    """
    First-fit decreasing packing with configurable sorting and placement strategy.
    Returns list of (box_idx, x, y, z, w, h, d).
    """
    n = len(boxes)
    
    # Sort boxes
    if sort_key is None:
        sort_key = lambda i, b: -b[0]*b[1]*b[2]  # Volume decreasing
    
    indexed_boxes = [(i, b) for i, b in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: sort_key(x[0], x[1]))
    
    placed = []  # (x, y, z, w, h, d)
    result = []  # (box_idx, x, y, z, w, h, d)
    
    for box_idx, (ow, oh, od) in indexed_boxes:
        if deadline is not None and time.time() > deadline:
            break
        
        # Try all orientations
        orientations = _all_orientations(ow, oh, od)
        
        best_pos = None
        best_orientation = None
        best_score = None
        
        for (w, h, d) in orientations:
            if w > W or h > H or d > D:
                continue
            
            pos = _find_best_position(w, h, d, W, H, D, placed, strategy=placement_strategy)
            if pos is not None:
                if placement_strategy == 'best_fit':
                    score = max(pos[0] + w, pos[1] + h, pos[2] + d)
                elif placement_strategy == 'min_gap':
                    score = (W - (pos[0] + w)) + (H - (pos[1] + h)) + (D - (pos[2] + d))
                else:
                    score = (pos[2], pos[1], pos[0])
                
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = pos
                    best_orientation = (w, h, d)
        
        if best_pos is not None:
            x, y, z = best_pos
            w, h, d = best_orientation
            placed.append((x, y, z, w, h, d))
            result.append((box_idx, x, y, z, w, h, d))
    
    return result

def _pack_shelf(boxes, W, H, D, sort_key=None, deadline=None):
    """
    Shelf-based packing. Each shelf has a fixed height.
    Returns list of (box_idx, x, y, z, w, h, d).
    """
    n = len(boxes)
    
    if sort_key is None:
        sort_key = lambda i, b: -max(b)  # Max dimension decreasing
    
    indexed_boxes = [(i, b) for i, b in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: sort_key(x[0], x[1]))
    
    placed = []  # (x, y, z, w, h, d)
    result = []  # (box_idx, x, y, z, w, h, d)
    
    # Shelf state
    shelf_x = 0
    shelf_y = 0
    shelf_h = 0
    layer_z = 0
    layer_d = 0
    
    for box_idx, (ow, oh, od) in indexed_boxes:
        if deadline is not None and time.time() > deadline:
            break
        
        # Choose orientation: minimize z-extent, then footprint
        orientations = _all_orientations(ow, oh, od)
        best_orient = min(orientations, key=lambda o: (o[2], max(o[0], o[1])))
        w, h, d = best_orient
        
        # Check if fits in current shelf
        if shelf_x + w > W:
            # Move to next row
            shelf_x = 0
            shelf_y += shelf_h
            shelf_h = 0
        
        if shelf_y + h > H:
            # Start new layer
            layer_z += layer_d
            layer_d = 0
            shelf_y = 0
            shelf_h = 0
            shelf_x = 0
        
        if layer_z + d > D:
            continue
        
        # Place box
        placed.append((shelf_x, shelf_y, layer_z, w, h, d))
        result.append((box_idx, shelf_x, shelf_y, layer_z, w, h, d))
        shelf_x += w
        shelf_h = max(shelf_h, h)
        layer_d = max(layer_d, d)
    
    return result

def _pack_layered(boxes, W, H, D, sort_key=None, deadline=None):
    """
    Layered packing: fill layers in z-direction, each layer is a 2D bin packing problem.
    Returns list of (box_idx, x, y, z, w, h, d).
    """
    n = len(boxes)
    
    if sort_key is None:
        sort_key = lambda i, b: -b[2]  # Depth (z-extent) decreasing
    
    indexed_boxes = [(i, b) for i, b in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: sort_key(x[0], x[1]))
    
    placed = []  # (x, y, z, w, h, d)
    result = []  # (box_idx, x, y, z, w, h, d)
    
    # Current layer state
    layer_z = 0
    layer_d = 0
    # 2D shelf state within current layer
    shelf_x = 0
    shelf_y = 0
    shelf_h = 0
    
    for box_idx, (ow, oh, od) in indexed_boxes:
        if deadline is not None and time.time() > deadline:
            break
        
        # Choose orientation: minimize z-extent (depth), then height
        orientations = _all_orientations(ow, oh, od)
        best_orient = min(orientations, key=lambda o: (o[2], o[1], o[0]))
        w, h, d = best_orient
        
        # Check if fits in current layer
        if layer_z + d > D:
            # Start new layer
            layer_z += layer_d
            layer_d = 0
            shelf_x = 0
            shelf_y = 0
            shelf_h = 0
        
        if layer_z + d > D:
            continue
        
        # 2D shelf packing in current layer
        if shelf_x + w > W:
            shelf_x = 0
            shelf_y += shelf_h
            shelf_h = 0
        
        if shelf_y + h > H:
            # Can't fit in this layer, try next layer
            continue
        
        # Place box
        placed.append((shelf_x, shelf_y, layer_z, w, h, d))
        result.append((box_idx, shelf_x, shelf_y, layer_z, w, h, d))
        shelf_x += w
        shelf_h = max(shelf_h, h)
        layer_d = max(layer_d, d)
    
    return result

def _make_output(placed, n):
    """Convert placed list to output format."""
    output = [(-1, 0, 0, 0, 0, 0, 0)] * n
    for (idx, x, y, z, w, h, d) in placed:
        if 0 <= idx < n:
            output[idx] = (0, x, y, z, w, h, d)
    return output

def _compute_volume(placed):
    """Compute total volume of placed boxes."""
    return sum(w * h * d for (_, _, _, _, w, h, d) in placed)

def _verify(placed, W, H, D):
    """Verify that placed boxes don't overlap and are within container."""
    for i, (idx, x, y, z, w, h, d) in enumerate(placed):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for j in range(i):
            jidx, jx, jy, jz, jw, jh, jd = placed[j]
            if not (x + w <= jx or jx + jw <= x or
                    y + h <= jy or jy + jh <= y or
                    z + d <= jz or jz + jd <= z):
                return False
    return True

def load(boxes, container):
    """
    boxes: list of (w, h, d) tuples
    container: (W, H, D) tuple
    returns: list of (bin, x, y, z, w, h, d) tuples
    """
    W, H, D = container
    n = len(boxes)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin (150s limit)
    
    best_result = None
    best_volume = 0
    
    # Define various sorting keys
    sort_keys = {
        'volume': lambda i, b: -b[0]*b[1]*b[2],
        'max_dim': lambda i, b: -max(b),
        'min_dim': lambda i, b: -min(b),
        'sum_dim': lambda i, b: -(b[0]+b[1]+b[2]),
        'aspect': lambda i, b: -(max(b) / (min(b) + 1e-9)),
        'height': lambda i, b: -b[1],
        'depth': lambda i, b: -b[2],
        'width': lambda i, b: -b[0],
    }
    
    placement_strategies = ['best_fit', 'first_fit', 'min_gap']
    
    # Try multiple combinations of strategies
    strategies_tried = set()
    
    # 1. First-fit decreasing with various sorts and placement strategies
    for sort_name, sort_key in sort_keys.items():
        for strat in placement_strategies:
            key = (sort_name, strat)
            if key in strategies_tried:
                continue
            strategies_tried.add(key)
            
            if time.time() > deadline - 5.0:
                break
            
            placed = _pack_first_fit_decreasing(
                boxes, W, H, D, 
                sort_key=sort_key, 
                placement_strategy=strat,
                deadline=deadline - 5.0
            )
            
            if not _verify(placed, W, H, D):
                continue
            
            vol = _compute_volume(placed)
            if vol > best_volume:
                best_volume = vol
                best_result = placed
    
    # 2. Shelf-based packing
    for sort_name, sort_key in sort_keys.items():
        if time.time() > deadline - 5.0:
            break
        
        placed = _pack_shelf(boxes, W, H, D, sort_key=sort_key, deadline=deadline - 5.0)
        
        if not _verify(placed, W, H, D):
            continue
        
        vol = _compute_volume(placed)
        if vol > best_volume:
            best_volume = vol
            best_result = placed
    
    # 3. Layered packing
    for sort_name, sort_key in sort_keys.items():
        if time.time() > deadline - 5.0:
            break
        
        placed = _pack_layered(boxes, W, H, D, sort_key=sort_key, deadline=deadline - 5.0)
        
        if not _verify(placed, W, H, D):
            continue
        
        vol = _compute_volume(placed)
        if vol > best_volume:
            best_volume = vol
            best_result = placed
    
    # If no valid result found, return all boxes as left out
    if best_result is None:
        return [(-1, 0, 0, 0, 0, 0, 0)] * n
    
    return _make_output(best_result, n)

# EVOLVE-BLOCK-END

