"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations

def _can_place(pos, dims, placed, W, H, D):
    x, y, z = pos
    w, h, d = dims
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if not (x + w <= px or px + pw <= x or
                y + h <= py or py + ph <= y or
                z + d <= pz or pz + pd <= z):
            return False
    return True

def _best_fit_pos(dims, placed, W, H, D):
    # Generate candidate positions: bottom-left-front corners derived from placed boxes
    # plus the origin. We only need to check positions where the new box's face is
    # aligned with an existing box's face or a container wall.
    # To keep it fast, we limit the number of candidates.
    candidates = set()
    candidates.add((0, 0, 0))
    
    # For each placed box, add positions adjacent to it
    for (px, py, pz, pw, ph, pd) in placed:
        w, h, d = dims
        # x candidates: left of box, right of box
        if px >= w: candidates.add((px - w, 0, 0))
        if px + pw + w <= W: candidates.add((px + pw, 0, 0))
        # y candidates: below box, above box
        if py >= h: candidates.add((0, py - h, 0))
        if py + ph + h <= H: candidates.add((0, py + ph, 0))
        # z candidates: in front of box, behind box
        if pz >= d: candidates.add((0, 0, pz - d))
        if pz + pd + d <= D: candidates.add((0, 0, pz + pd))
        
        # Also combinations of x and y, x and z, y and z
        if px >= w and py >= h: candidates.add((px - w, py - h, 0))
        if px >= w and py + ph + h <= H: candidates.add((px - w, py + ph, 0))
        if px + pw + w <= W and py >= h: candidates.add((px + pw, py - h, 0))
        if px + pw + w <= W and py + ph + h <= H: candidates.add((px + pw, py + ph, 0))
        
        if px >= w and pz >= d: candidates.add((px - w, 0, pz - d))
        if px >= w and pz + pd + d <= D: candidates.add((px - w, 0, pz + pd))
        if px + pw + w <= W and pz >= d: candidates.add((px + pw, 0, pz - d))
        if px + pw + w <= W and pz + pd + d <= D: candidates.add((px + pw, 0, pz + pd))
        
        if py >= h and pz >= d: candidates.add((0, py - h, pz - d))
        if py >= h and pz + pd + d <= D: candidates.add((0, py - h, pz + pd))
        if py + ph + h <= H and pz >= d: candidates.add((0, py + ph, pz - d))
        if py + ph + h <= H and pz + pd + d <= D: candidates.add((0, py + ph, pz + pd))
        
        # Full 3D corners
        for cx in [px - w, px + pw]:
            for cy in [py - h, py + ph]:
                for cz in [pz - d, pz + pd]:
                    if 0 <= cx and cx + w <= W and 0 <= cy and cy + h <= H and 0 <= cz and cz + d <= D:
                        candidates.add((cx, cy, cz))
    
    # Find the best position (minimize x, then y, then z for compactness)
    best_pos = None
    best_key = None
    for pos in candidates:
        if _can_place(pos, dims, placed, W, H, D):
            key = (pos[0], pos[1], pos[2])
            if best_key is None or key < best_key:
                best_key = key
                best_pos = pos
    return best_pos

def _solve_order(boxes, container, order):
    W, H, D = container
    placed = []
    result = [None] * len(boxes)
    total_vol = 0
    
    for idx in order:
        w, h, d = boxes[idx]
        best_pos = _best_fit_pos((w, h, d), placed, W, H, D)
        if best_pos is not None:
            x, y, z = best_pos
            placed.append((x, y, z, w, h, d))
            result[idx] = (0, x, y, z, w, h, d)
            total_vol += w * h * d
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    return result, total_vol

def load(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)
    
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Pre-compute all possible rotations for each box
    # We'll try all 6 permutations of (w, h, d) for each box
    rotations = []
    for w, h, d in boxes:
        rots = list(set(permutations([w, h, d])))
        rotations.append(rots)
    
    # Generate multiple sorting strategies
    strategies = []
    
    # Strategy 1: Sort by volume descending
    vol_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    strategies.append(vol_order)
    
    # Strategy 2: Sort by max dimension descending
    max_dim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    strategies.append(max_dim_order)
    
    # Strategy 3: Sort by min dimension descending
    min_dim_order = sorted(range(n), key=lambda i: min(boxes[i]), reverse=True)
    strategies.append(min_dim_order)
    
    # Strategy 4: Sort by base area (w*h) descending
    base_area_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1], reverse=True)
    strategies.append(base_area_order)
    
    # Strategy 5: Sort by longest edge first
    longest_edge_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    strategies.append(longest_edge_order)
    
    # Strategy 6: Sort by volume ascending (try small first)
    vol_asc_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2])
    strategies.append(vol_asc_order)
    
    # Strategy 7: Sort by max dimension ascending
    max_dim_asc_order = sorted(range(n), key=lambda i: max(boxes[i]))
    strategies.append(max_dim_asc_order)
    
    # Strategy 8: Random order
    random_order = list(range(n))
    random.shuffle(random_order)
    strategies.append(random_order)
    
    # Strategy 9: Sort by aspect ratio (max/min) descending
    aspect_order = sorted(range(n), key=lambda i: max(boxes[i])/max(min(boxes[i]),1), reverse=True)
    strategies.append(aspect_order)
    
    # Strategy 10: Sort by aspect ratio ascending
    aspect_asc_order = sorted(range(n), key=lambda i: max(boxes[i])/max(min(boxes[i]),1))
    strategies.append(aspect_asc_order)
    
    best_result = None
    best_vol = 0
    
    for order in strategies:
        if time.time() > deadline:
            break
        
        # Try the order with best-fit placement
        result, total_vol = _solve_order(boxes, container, order)
        
        if total_vol > best_vol:
            best_vol = total_vol
            best_result = result
    
    # If we have time, try random restarts with shuffled orders
    while time.time() < deadline:
        random_order = list(range(n))
        random.shuffle(random_order)
        result, total_vol = _solve_order(boxes, container, random_order)
        if total_vol > best_vol:
            best_vol = total_vol
            best_result = result
    
    if best_result is None:
        # Fallback: simple shelf packing
        placements = []
        x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
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
    
    return best_result
# EVOLVE-BLOCK-END
