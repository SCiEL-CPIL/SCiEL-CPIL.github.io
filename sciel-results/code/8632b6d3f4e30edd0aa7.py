import time
import random
from itertools import permutations

def _rotations(w, h, d):
    seen = set()
    result = []
    for perm in permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits(x, y, z, bw, bh, bd, W, H, D, placed):
    if x + bw > W or y + bh > H or z + bd > D:
        return False
    for _, px, py, pz, pw, ph, pd in placed:
        if x < px + pw and px < x + bw and \
           y < py + ph and py < y + bh and \
           z < pz + pd and pz < z + bd:
            return False
    return True

def _skyline_greedy(boxes, container, order, time_start, timeout):
    """
    2D floor-plan projection with vertical skyline columns.
    We discretize X and Y into candidate positions (extreme points from placed boxes).
    For each candidate (x, y), we maintain a skyline of Z-intervals.
    When placing a box, we check all candidate (x,y) and find the lowest valid Z.
    """
    W, H, D = container
    box_rots = [_rotations(w, h, d) for w, h, d in boxes]
    
    # Extreme point generation
    Xs = {0}
    Ys = {0}
    placed = []  # (idx, x, y, z, w, h, d)
    total_vol = 0
    
    for i in order:
        if time.time() - time_start > timeout:
            break
        w, h, d = boxes[i]
        rots = box_rots[i]
        
        best_pos = None
        best_cost = float('inf')
        
        for bw, bh, bd in rots:
            for x in Xs:
                if x + bw > W:
                    continue
                for y in Ys:
                    if y + bh > H:
                        continue
                    
                    # Find min valid z using skyline logic
                    min_z = 0
                    for _, px, py, pz, pw, ph, pd in placed:
                        if x < px + pw and px < x + bw and \
                           y < py + ph and py < y + bh:
                            if pz + pd > min_z:
                                min_z = pz + pd
                    
                    if min_z + bd <= D:
                        # Cost: prefer lower z, then lower x, then lower y
                        # This minimizes vertical fragmentation
                        cost = min_z * 10000 + x * 100 + y
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = (x, y, min_z, bw, bh, bd)
        
        if best_pos:
            placed.append((i, best_pos[0], best_pos[1], best_pos[2], best_pos[3], best_pos[4], best_pos[5]))
            total_vol += best_pos[3] * best_pos[4] * best_pos[5]
            # Update extreme points
            Xs.add(best_pos[0] + best_pos[3])
            Ys.add(best_pos[1] + best_pos[4])
    
    return placed, total_vol

def _shelf_greedy(boxes, container, order, time_start, timeout):
    """
    Shelf-based packing: place boxes on horizontal shelves.
    Each shelf has a height, and boxes are placed left-to-right on the shelf.
    When a box doesn't fit on the current shelf, start a new shelf above.
    """
    W, H, D = container
    box_rots = [_rotations(w, h, d) for w, h, d in boxes]
    
    placed = []  # (idx, x, y, z, w, h, d)
    total_vol = 0
    shelves = []  # list of (shelf_z, shelf_h, current_x)
    
    for i in order:
        if time.time() - time_start > timeout:
            break
        w, h, d = boxes[i]
        rots = box_rots[i]
        
        best_pos = None
        best_cost = float('inf')
        
        for bw, bh, bd in rots:
            # Try placing on each existing shelf
            for shelf_idx, (sz, sh, cx) in enumerate(shelves):
                if bh > H - sz:
                    continue
                if cx + bw <= W and bd <= D:
                    # Valid position on this shelf
                    cost = sz * 10000 + cx * 100 + bd
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (cx, sz, 0, bw, bh, bd, shelf_idx)
            
            # Try starting a new shelf
            # Find the lowest position where a new shelf of height bh fits
            new_shelf_z = 0
            for sz, sh, cx in shelves:
                if sz > new_shelf_z:
                    new_shelf_z = sz
            # Actually, we need to find the max bottom of existing shelves
            # Let's re-think: new shelf goes on top of the tallest stack
            max_top = 0
            for sz, sh, cx in shelves:
                if sz + sh > max_top:
                    max_top = sz + sh
            
            if max_top + bh <= H and bw <= W and bd <= D:
                cost = max_top * 10000 + 0 * 100 + bd
                if cost < best_cost:
                    best_cost = cost
                    best_pos = (0, max_top, 0, bw, bh, bd, -1)  # -1 indicates new shelf
        
        if best_pos:
            x, y, z, bw, bh, bd, shelf_idx = best_pos
            # y in our representation is the z-coordinate (height)
            placed.append((i, x, y, z, bw, bh, bd))
            total_vol += bw * bh * bd
            
            if shelf_idx >= 0:
                # Update existing shelf's current x
                sz, sh, cx = shelves[shelf_idx]
                shelves[shelf_idx] = (sz, sh, x + bw)
            else:
                # Add new shelf
                shelves.append((y, bh, x + bw))
    
    return placed, total_vol

def _extreme_point_greedy(boxes, container, order, time_start, timeout):
    """
    Extreme Points Approach: maintain a list of extreme points where boxes can be placed.
    """
    W, H, D = container
    box_rots = [_rotations(w, h, d) for w, h, d in boxes]
    
    # Extreme points: candidate positions where a box's corner can be placed
    extreme_points = {(0, 0, 0)}
    placed = []
    total_vol = 0
    
    for i in order:
        if time.time() - time_start > timeout:
            break
        w, h, d = boxes[i]
        rots = box_rots[i]
        
        best_pos = None
        best_cost = float('inf')
        
        for bw, bh, bd in rots:
            for ex, ey, ez in list(extreme_points):
                if ex + bw > W or ey + bh > H or ez + bd > D:
                    continue
                
                # Check if this position is valid (no overlap)
                valid = True
                for _, px, py, pz, pw, ph, pd in placed:
                    if ex < px + pw and px < ex + bw and \
                       ey < py + ph and py < ey + bh and \
                       ez < pz + pd and pz < ez + bd:
                        valid = False
                        break
                
                if valid:
                    cost = ez * 10000 + ex * 100 + ey
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (ex, ey, ez, bw, bh, bd)
        
        if best_pos:
            x, y, z, bw, bh, bd = best_pos
            placed.append((i, x, y, z, bw, bh, bd))
            total_vol += bw * bh * bd
            
            # Generate new extreme points
            # From the placed box, new extreme points are:
            # (x+bw, y, z), (x, y+bh, z), (x, y, z+bd)
            # And combinations with existing extreme points
            new_points = set()
            new_points.add((x + bw, y, z))
            new_points.add((x, y + bh, z))
            new_points.add((x, y, z + bd))
            new_points.add((x + bw, y + bh, z))
            new_points.add((x + bw, y, z + bd))
            new_points.add((x, y + bh, z + bd))
            new_points.add((x + bw, y + bh, z + bd))
            extreme_points.update(new_points)
    
    return placed, total_vol

def _local_search(placed, boxes, container, time_start, timeout):
    """
    Simple local search: try to move placed boxes to better positions.
    """
    W, H, D = container
    total_vol = sum(p[4]*p[5]*p[6] for p in placed)
    
    # Try swapping boxes
    # This is a simplified version; full local search would be more complex
    # For now, we just return the current placement
    return placed, total_vol

def load(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    
    start_time = time.time()
    timeout = 140.0  # Safety margin for 150s limit
    
    # Pre-compute rotations
    box_rots = [_rotations(w, h, d) for w, h, d in boxes]
    
    # Multiple sorting strategies
    indices = list(range(len(boxes)))
    
    strategies = {
        'volume_desc': sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True),
        'max_dim_desc': sorted(indices, key=lambda i: max(boxes[i]), reverse=True),
        'surface_area_desc': sorted(indices, key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True),
        'min_dim_asc': sorted(indices, key=lambda i: min(boxes[i])),
    }
    
    # Algorithms
    algorithms = {
        'skyline': _skyline_greedy,
        'shelf': _shelf_greedy,
        'extreme_point': _extreme_point_greedy,
    }
    
    best_vol = 0
    best_placed = []
    
    # Try all combinations of algorithms and strategies
    for alg_name, alg_func in algorithms.items():
        for strat_name, order in strategies.items():
            if time.time() - start_time > timeout:
                break
            
            placed, vol = alg_func(boxes, container, order, start_time, timeout)
            if vol > best_vol:
                best_vol = vol
                best_placed = placed
    
    # Construct output
    result = [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]
    
    for p in best_placed:
        idx = p[0]
        x, y, z, w, h, d = p[1], p[2], p[3], p[4], p[5], p[6]
        # Validate bounds
        if 0 <= x and x + w <= W and 0 <= y and y + h <= H and 0 <= z and z + d <= D:
            result[idx] = (0, x, y, z, w, h, d)
    
    return result
