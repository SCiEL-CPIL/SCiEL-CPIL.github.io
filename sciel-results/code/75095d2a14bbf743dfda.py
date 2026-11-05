"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import permutations

def _get_rotations(w, h, d):
    """Get all unique rotations of a box."""
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    return list(rots)

def _find_position(placed, bw, bh, bd, W, H, D):
    """Find a valid position for a box of dimensions (bw, bh, bd)."""
    # Try to place at the lowest possible z, then y, then x
    for z in range(0, D - bd + 1):
        for y in range(0, H - bh + 1):
            for x in range(0, W - bw + 1):
                valid = True
                for px, py, pz, pw, ph, pd in placed:
                    if (x < px + pw and x + bw > px and
                        y < py + ph and y + bh > py and
                        z < pz + pd and z + bd > pz):
                        valid = False
                        break
                if valid:
                    return (x, y, z)
    return None

def _try_place(boxes, container, seed=0):
    """
    Attempt to place boxes into the container.
    Returns a list of placements: (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        # Sort rotations by volume (all same) then by height (prefer flatter)
        rots.sort(key=lambda r: (r[1], r[2], r[0]))
        box_rotations.append(rots)
    
    # Sort boxes by volume (largest first) for better packing
    box_indices = list(range(n))
    box_volumes = [w * h * d for w, h, d in boxes]
    box_indices.sort(key=lambda i: -box_volumes[i])
    
    placed = []  # List of (x, y, z, w, h, d)
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    filled_volume = 0
    
    for idx in box_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        rots = box_rotations[idx]
        
        placed_this = False
        for bw, bh, bd in rots:
            if time.time() > deadline:
                break
            
            pos = _find_position(placed, bw, bh, bd, W, H, D)
            if pos is not None:
                x, y, z = pos
                placed.append((x, y, z, bw, bh, bd))
                result[idx] = (0, x, y, z, bw, bh, bd)
                filled_volume += bw * bh * bd
                placed_this = True
                break
        
        if not placed_this:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    return result

def _try_place_shelf(boxes, container):
    """Shelf-based packing: place boxes in shelves/layers."""
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        rots.sort(key=lambda r: (r[1], r[2], r[0]))
        box_rotations.append(rots)
    
    box_indices = list(range(n))
    box_volumes = [w * h * d for w, h, d in boxes]
    box_indices.sort(key=lambda i: -box_volumes[i])
    
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    filled_volume = 0
    
    # Shelf-based: track shelves as (y_start, y_end, x_cursor, height)
    shelves = []  # Each shelf: [y, height, x_cursor]
    current_z = 0
    current_d = 0
    
    for idx in box_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        rots = box_rotations[idx]
        
        placed_this = False
        for bw, bh, bd in rots:
            if time.time() > deadline:
                break
            
            # Try to place in existing shelf
            for shelf in shelves:
                y_start, shelf_h, x_cur = shelf
                if bh <= shelf_h and x_cur + bw <= W:
                    z_pos = current_z
                    if z_pos + bd <= D:
                        result[idx] = (0, x_cur, y_start, z_pos, bw, bh, bd)
                        shelf[2] = x_cur + bw
                        filled_volume += bw * bh * bd
                        placed_this = True
                        break
            
            if placed_this:
                continue
            
            # Try to create new shelf at current layer
            if current_z + bd <= D:
                # Find a shelf that can accommodate this box's height
                best_shelf = None
                for i, shelf in enumerate(shelves):
                    y_start, shelf_h, x_cur = shelf
                    if bh <= shelf_h:
                        if best_shelf is None or x_cur < shelves[best_shelf][2]:
                            best_shelf = i
                
                if best_shelf is not None and shelves[best_shelf][2] + bw <= W:
                    y_start = shelves[best_shelf][1]
                    z_pos = current_z
                    if z_pos + bd <= D:
                        result[idx] = (0, shelves[best_shelf][2], y_start, z_pos, bw, bh, bd)
                        shelves[best_shelf][2] += bw
                        filled_volume += bw * bh * bd
                        placed_this = True
                elif not shelves or shelves[-1][2] + bw > W:
                    # Start new shelf
                    if current_z + bd <= D:
                        result[idx] = (0, 0, 0, current_z, bw, bh, bd)
                        shelves.append([0, bh, bw])
                        filled_volume += bw * bh * bd
                        placed_this = True
        
        if not placed_this:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    return result

def _try_place_feasible(boxes, container):
    """Feasible function based packing - try to fit boxes in corners."""
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        rots.sort(key=lambda r: (r[1], r[2], r[0]))
        box_rotations.append(rots)
    
    box_indices = list(range(n))
    box_volumes = [w * h * d for w, h, d in boxes]
    box_indices.sort(key=lambda i: -box_volumes[i])
    
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    filled_volume = 0
    placed = []
    
    for idx in box_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        rots = box_rotations[idx]
        
        placed_this = False
        for bw, bh, bd in rots:
            if time.time() > deadline:
                break
            
            # Try to place at the "lowest" available position
            # Check if box fits in remaining space
            can_place = True
            best_pos = None
            
            # Try placing at origin first, then shift
            for z in range(0, D - bd + 1):
                for y in range(0, H - bh + 1):
                    for x in range(0, W - bw + 1):
                        valid = True
                        for px, py, pz, pw, ph, pd in placed:
                            if (x < px + pw and x + bw > px and
                                y < py + ph and y + bh > py and
                                z < pz + pd and z + bd > pz):
                                valid = False
                                break
                        if valid:
                            best_pos = (x, y, z)
                            break
                    if best_pos:
                        break
                if best_pos:
                    break
            
            if best_pos:
                x, y, z = best_pos
                placed.append((x, y, z, bw, bh, bd))
                result[idx] = (0, x, y, z, bw, bh, bd)
                filled_volume += bw * bh * bd
                placed_this = True
                break
        
        if not placed_this:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    return result

def _evaluate_result(placement, container):
    """Evaluate the fraction of container volume filled."""
    W, H, D = container
    total_volume = W * H * D
    if total_volume == 0:
        return 0.0
    
    filled = 0
    for bin_id, x, y, z, w, h, d in placement:
        if bin_id == 0:
            # Check if placement is valid
            if x >= 0 and y >= 0 and z >= 0 and \
               x + w <= W and y + h <= H and z + d <= D:
                filled += w * h * d
    
    return filled / total_volume

def load(boxes, container):
    """
    Main entry point: place boxes into container.
    Returns list of (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Try multiple strategies and keep the best
    best_placement = None
    best_score = -1
    
    # Strategy 1: Simple shelf packing
    placement1 = _try_place_shelf(boxes, container)
    score1 = _evaluate_result(placement1, container)
    if score1 > best_score:
        best_score = score1
        best_placement = placement1
    
    if time.time() > deadline:
        return best_placement
    
    # Strategy 2: Feasible/corner-based packing
    placement2 = _try_place_feasible(boxes, container)
    score2 = _evaluate_result(placement2, container)
    if score2 > best_score:
        best_score = score2
        best_placement = placement2
    
    if time.time() > deadline:
        return best_placement
    
    # Strategy 3: General placement with rotation search
    placement3 = _try_place(boxes, container)
    score3 = _evaluate_result(placement3, container)
    if score3 > best_score:
        best_score = score3
        best_placement = placement3
    
    return best_placement
# EVOLVE-BLOCK-END
