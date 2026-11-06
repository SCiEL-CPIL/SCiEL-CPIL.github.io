"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import product

def _get_rotations(w, h, d):
    """Get all unique rotations of a box."""
    rots = set()
    rots.add((w, h, d))
    rots.add((w, d, h))
    rots.add((h, w, d))
    rots.add((h, d, w))
    rots.add((d, w, h))
    rots.add((d, h, w))
    return list(rots)

def _check_overlap(x, y, z, w, h, d, placed):
    """Check if a box at (x,y,z) with dims (w,h,d) overlaps any placed box."""
    for px, py, pz, pw, ph, pd in placed:
        if not (x + w <= px or px + pw <= x or
                y + h <= py or py + ph <= y or
                z + d <= pz or pz + pd <= z):
            return True
    return False

def _fits_in_container(x, y, z, w, h, d, W, H, D):
    """Check if box fits within container boundaries."""
    return (x >= 0 and y >= 0 and z >= 0 and
            x + w <= W and y + h <= H and z + d <= D)

def _generate_candidates(placed, W, H, D):
    """Generate candidate positions for placing a new box."""
    candidates = [(0, 0, 0)]
    for px, py, pz, pw, ph, pd in placed:
        candidates.append((px + pw, py, pz))
        candidates.append((px, py + ph, pz))
        candidates.append((px, py, pz + pd))
        candidates.append((px + pw, py + ph, pz))
        candidates.append((px + pw, py, pz + pd))
        candidates.append((px, py + ph, pz + pd))
        candidates.append((px + pw, py + ph, pz + pd))
    
    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique

def _try_place(boxes, container, seed=0, strategy='best_fit'):
    """
    Attempt to place boxes into the container.
    Returns a list of placements: (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Preprocess
    box_rotations = []
    volumes = []
    for w, h, d in boxes:
        box_rotations.append(_get_rotations(w, h, d))
        volumes.append(w * h * d)
    
    container_vol = W * H * D
    if container_vol == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    def _check_time():
        return time.time() < deadline
    
    def _score_result(placement_result):
        """Calculate filled volume fraction."""
        filled = sum(w * h * d for b, x, y, z, w, h, d in placement_result if b == 0)
        return filled / container_vol
    
    def _place_with_order(order, rot_strategy='best', pos_strategy='corner'):
        """
        Place boxes in given order.
        rot_strategy: 'original', 'best', 'min_corner'
        pos_strategy: 'corner', 'lowest'
        """
        if not _check_time():
            return None
        
        placed = []
        result = [None] * n
        
        for idx in order:
            if not _check_time():
                return None
            
            ow, oh, od = boxes[idx]
            rots = box_rotations[idx]
            
            if rot_strategy == 'original':
                rots_to_try = [(ow, oh, od)]
            else:
                rots_to_try = rots
            
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            for rw, rh, rd in rots_to_try:
                candidates = _generate_candidates(placed, W, H, D)
                
                if pos_strategy == 'lowest':
                    candidates.sort(key=lambda p: (p[2], p[1], p[0]))
                else:
                    candidates.sort(key=lambda p: (p[2], p[1], p[0]))
                
                for cx, cy, cz in candidates:
                    if _fits_in_container(cx, cy, cz, rw, rh, rd, W, H, D):
                        if not _check_overlap(cx, cy, cz, rw, rh, rd, placed):
                            if rot_strategy == 'best':
                                score = cx + cy + cz
                            elif rot_strategy == 'min_corner':
                                score = cx + cy + cz
                            else:
                                score = 0
                            
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, cz)
                                best_rot = (rw, rh, rd)
                            break  # First valid position for this rotation
            
            if best_pos is not None:
                x, y, z = best_pos
                w, h, d = best_rot
                result[idx] = (0, x, y, z, w, h, d)
                placed.append((x, y, z, w, h, d))
            else:
                result[idx] = (-1, 0, 0, 0, ow, oh, od)
        
        # Fill in any None entries
        for i in range(n):
            if result[i] is None:
                ow, oh, od = boxes[i]
                result[i] = (-1, 0, 0, 0, ow, oh, od)
        
        return result
    
    # Multiple strategies
    best_result = None
    best_score = 0.0
    
    # Strategy 1: Volume descending
    order1 = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    if _check_time():
        r = _place_with_order(order1, 'best', 'corner')
        if r is not None:
            s = _score_result(r)
            if s > best_score:
                best_score = s
                best_result = r
    
    # Strategy 2: Max dimension descending
    order2 = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    if _check_time():
        r = _place_with_order(order2, 'best', 'corner')
        if r is not None:
            s = _score_result(r)
            if s > best_score:
                best_score = s
                best_result = r
    
    # Strategy 3: Min dimension ascending
    order3 = sorted(range(n), key=lambda i: min(boxes[i]))
    if _check_time():
        r = _place_with_order(order3, 'best', 'corner')
        if r is not None:
            s = _score_result(r)
            if s > best_score:
                best_score = s
                best_result = r
    
    # Strategy 4: Random restarts
    rng = random.Random(seed)
    max_restarts = 30
    for r in range(max_restarts):
        if not _check_time():
            break
        order_rand = list(range(n))
        rng.shuffle(order_rand)
        res = _place_with_order(order_rand, 'best', 'corner')
        if res is not None:
            s = _score_result(res)
            if s > best_score:
                best_score = s
                best_result = res
    
    # Strategy 5: Layer-based packing
    if _check_time():
        best_layer_result = _try_layer_packing(boxes, container, volumes, box_rotations, deadline)
        if best_layer_result is not None:
            s = _score_result(best_layer_result)
            if s > best_score:
                best_score = s
                best_result = best_layer_result
    
    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_result

def _try_layer_packing(boxes, container, volumes, box_rotations, deadline):
    """
    Time-sliced layer decomposition: discretize container height into slabs,
    pack boxes into each slab using 2D shelf heuristic.
    """
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    start_time = time.time()
    
    def _check_time():
        return time.time() < deadline - 5.0
    
    # Sort boxes by volume descending
    order = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    
    # Try different layer heights
    best_result = None
    best_score = 0.0
    container_vol = W * H * D
    
    # Generate candidate layer heights based on box dimensions
    layer_heights = set()
    layer_heights.add(H)  # Single layer
    layer_heights.add(H // 2)
    layer_heights.add(H // 3)
    
    # Add box heights as potential layer heights
    for i in range(n):
        for rw, rh, rd in box_rotations[i]:
            if rh > 0 and rh <= H:
                layer_heights.add(rh)
            if rh > 0 and rh <= H // 2:
                layer_heights.add(rh)
    
    # Filter valid layer heights
    layer_heights = sorted([lh for lh in layer_heights if lh > 0 and lh <= H], reverse=True)
    
    # Limit number of layer heights to try
    layer_heights = layer_heights[:15]
    
    for layer_h in layer_heights:
        if not _check_time():
            break
        
        placed = []
        result = [None] * n
        current_z = 0
        
        # Process boxes in order
        for idx in order:
            if not _check_time():
                return None
            
            ow, oh, od = boxes[idx]
            rots = box_rotations[idx]
            
            # Find a rotation and position
            best_pos = None
            best_rot = None
            best_score_local = float('inf')
            
            for rw, rh, rd in rots:
                # Check if this rotation fits in current layer
                if rh > layer_h:
                    continue
                
                # Try to place in current layer using 2D shelf
                # Generate candidate x,y positions
                candidates = [(0, 0)]
                for px, py, pz, pw, ph, pd in placed:
                    if pz + pd <= current_z + layer_h and current_z <= pz + pd:
                        # In same layer
                        candidates.append((px + pw, py))
                        candidates.append((px, py + ph))
                        candidates.append((px + pw, py + ph))
                
                # Deduplicate
                seen = set()
                unique_cands = []
                for cx, cy in candidates:
                    if (cx, cy) not in seen:
                        seen.add((cx, cy))
                        unique_cands.append((cx, cy))
                
                # Sort by y then x (prefer lower, more compact)
                unique_cands.sort(key=lambda p: (p[1], p[0]))
                
                for cx, cy in unique_cands:
                    # Check bounds
                    if cx + rw > W or cy + rd > D:
                        continue
                    
                    # Check overlap with placed boxes in this layer
                    overlap = False
                    for px, py, pz, pw, ph, pd in placed:
                        # Check if in same layer
                        if pz + pd <= current_z or current_z + layer_h <= pz:
                            continue
                        # 2D overlap check
                        if not (cx + rw <= px or px + pw <= cx or
                                cy + rd <= py or py + ph <= cy):
                            overlap = True
                            break
                    
                    if not overlap:
                        score = cy + cx  # Prefer compact
                        if score < best_score_local:
                            best_score_local = score
                            best_pos = (cx, cy, current_z)
                            best_rot = (rw, rh, rd)
                        break
            
            if best_pos is not None:
                x, y, z = best_pos
                w, h, d = best_rot
                result[idx] = (0, x, y, z, w, h, d)
                placed.append((x, y, z, w, h, d))
            else:
                # Try to start a new layer
                new_z = current_z + layer_h
                if new_z + layer_h <= H:
                    # Check if any rotation fits in a new layer
                    for rw, rh, rd in rots:
                        if rh > layer_h:
                            continue
                        # Try at origin of new layer
                        if rw <= W and rd <= D:
                            # Check no overlap (should be fine since new layer)
                            result[idx] = (0, 0, 0, new_z, rw, rh, rd)
                            placed.append((0, 0, new_z, rw, rh, rd))
                            current_z = new_z
                            break
                    else:
                        result[idx] = (-1, 0, 0, 0, ow, oh, od)
                else:
                    result[idx] = (-1, 0, 0, 0, ow, oh, od)
        
        # Fill in any None entries
        for i in range(n):
            if result[i] is None:
                ow, oh, od = boxes[i]
                result[i] = (-1, 0, 0, 0, ow, oh, od)
        
        # Score
        filled = sum(w * h * d for b, x, y, z, w, h, d in result if b == 0)
        score = filled / container_vol if container_vol > 0 else 0
        
        if score > best_score:
            best_score = score
            best_result = result
    
    return best_result

def _try_place_diverse(boxes, container, seed=0):
    """Try diverse strategies and return the best result."""
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    def _check_time():
        return time.time() < deadline
    
    # Get rotations
    box_rotations = []
    volumes = []
    for w, h, d in boxes:
        box_rotations.append(_get_rotations(w, h, d))
        volumes.append(w * h * d)
    
    container_vol = W * H * D
    if container_vol == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    def _score_result(placement_result):
        filled = sum(w * h * d for b, x, y, z, w, h, d in placement_result if b == 0)
        return filled / container_vol
    
    best_result = None
    best_score = 0.0
    
    # Multiple sorting strategies
    strategies = [
        sorted(range(n), key=lambda i: volumes[i], reverse=True),
        sorted(range(n), key=lambda i: max(boxes[i]), reverse=True),
        sorted(range(n), key=lambda i: min(boxes[i])),
        sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1], reverse=True),
        sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][2], reverse=True),
        sorted(range(n), key=lambda i: boxes[i][1] * boxes[i][2], reverse=True),
    ]
    
    for order in strategies:
        if not _check_time():
            break
        
        placed = []
        result = [None] * n
        
        for idx in order:
            if not _check_time():
                return None
            
            ow, oh, od = boxes[idx]
            rots = box_rotations[idx]
            
            best_pos = None
            best_rot = None
            best_score_local = float('inf')
            
            for rw, rh, rd in rots:
                candidates = _generate_candidates(placed, W, H, D)
                candidates.sort(key=lambda p: (p[2], p[1], p[0]))
                
                for cx, cy, cz in candidates:
                    if _fits_in_container(cx, cy, cz, rw, rh, rd, W, H, D):
                        if not _check_overlap(cx, cy, cz, rw, rh, rd, placed):
                            score = cx + cy + cz
                            if score < best_score_local:
                                best_score_local = score
                                best_pos = (cx, cy, cz)
                                best_rot = (rw, rh, rd)
                            break
            
            if best_pos is not None:
                x, y, z = best_pos
                w, h, d = best_rot
                result[idx] = (0, x, y, z, w, h, d)
                placed.append((x, y, z, w, h, d))
            else:
                result[idx] = (-1, 0, 0, 0, ow, oh, od)
        
        for i in range(n):
            if result[i] is None:
                ow, oh, od = boxes[i]
                result[i] = (-1, 0, 0, 0, ow, oh, od)
        
        s = _score_result(result)
        if s > best_score:
            best_score = s
            best_result = result
    
    # Random restarts
    rng = random.Random(seed)
    for r in range(20):
        if not _check_time():
            break
        
        order_rand = list(range(n))
        rng.shuffle(order_rand)
        
        placed = []
        result = [None] * n
        
        for idx in order_rand:
            if not _check_time():
                break
            
            ow, oh, od = boxes[idx]
            rots = box_rotations[idx]
            
            best_pos = None
            best_rot = None
            best_score_local = float('inf')
            
            for rw, rh, rd in rots:
                candidates = _generate_candidates(placed, W, H, D)
                candidates.sort(key=lambda p: (p[2], p[1], p[0]))
                
                for cx, cy, cz in candidates:
                    if _fits_in_container(cx, cy, cz, rw, rh, rd, W, H, D):
                        if not _check_overlap(cx, cy, cz, rw, rh, rd, placed):
                            score = cx + cy + cz
                            if score < best_score_local:
                                best_score_local = score
                                best_pos = (cx, cy, cz)
                                best_rot = (rw, rh, rd)
                            break
            
            if best_pos is not None:
                x, y, z = best_pos
                w, h, d = best_rot
                result[idx] = (0, x, y, z, w, h, d)
                placed.append((x, y, z, w, h, d))
            else:
                result[idx] = (-1, 0, 0, 0, ow, oh, od)
        
        for i in range(n):
            if result[i] is None:
                ow, oh, od = boxes[i]
                result[i] = (-1, 0, 0, 0, ow, oh, od)
        
        s = _score_result(result)
        if s > best_score:
            best_score = s
            best_result = result
    
    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_result

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Try the main algorithm
    try:
        result = _try_place(boxes, container, seed=42)
        return result
    except Exception:
        # Fallback to simple greedy
        placements = []
        x, y, z = 0, 0, 0
        row_h, layer_d = 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements
# EVOLVE-BLOCK-END
