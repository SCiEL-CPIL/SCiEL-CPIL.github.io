<<<<<<< SEARCH
"""3D Container Loading: combined extreme-point greedy + multi-heuristic + local search."""
import time
import random
import itertools
import sys
=======
"""3D Container Loading: combined extreme-point greedy + multi-heuristic + local search."""
import time
import random
import itertools
import sys
import numpy as np
>>>>>>> REPLACE
<<<<<<< SEARCH
def _rotations(w, h, d):
    """Return all unique orientations of a box (w,h,d)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result
=======
def _rotations(w, h, d):
    """Return all unique orientations of a box (w,h,d)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _build_grid(container, placed):
    """Build a 3D occupancy grid from placed boxes. Returns (grid, W, H, D)."""
    W, H, D = container
    grid = np.zeros((W, H, D), dtype=np.int8)
    for (px, py, pz, pw, ph, pd) in placed:
        x1, x2 = int(px), int(px + pw)
        y1, y2 = int(py), int(py + ph)
        z1, z2 = int(pz), int(pz + pd)
        grid[x1:x2, y1:y2, z1:z2] = 1
    return grid, W, H, D

def _fits_grid(grid, x, y, z, w, h, d):
    """Check if box fits at (x,y,z) with dims (w,h,d) using grid."""
    x1, x2 = int(x), int(x + w)
    y1, y2 = int(y), int(y + h)
    z1, z2 = int(z), int(z + d)
    if x2 > grid.shape[0] or y2 > grid.shape[1] or z2 > grid.shape[2]:
        return False
    return not np.any(grid[x1:x2, y1:y2, z1:z2])
>>>>>>> REPLACE
<<<<<<< SEARCH
def _greedy(boxes, container, order, deadline, pos_strategy, score_strategy):
    """Greedy placement with given position and scoring strategies."""
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    total_vol = 0.0
    n = len(boxes)
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        if pos_strategy == 'extreme':
            candidates = _extreme_points(placed, container)
            if (0, 0, 0) not in candidates:
                candidates.append((0, 0, 0))
        else:  # faces
            candidates = _face_points(placed, container)
        
        best_pos = None
        best_rot = None
        best_key = None
        
        for pos in candidates:
            for rw, rh, rd in rots:
                if _fits((rw, rh, rd), pos, container, placed):
                    if score_strategy == 'compact':
                        key = _score_position_compact(pos, (rw, rh, rd), placed, container)
                    else:
                        key = _score_position_low(pos, (rw, rh, rd), placed, container)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = pos
                        best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_rot
            placed.append((x, y, z, rw, rh, rd))
            placements[idx] = (0, x, y, z, rw, rh, rd)
            total_vol += rw * rh * rd
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements, total_vol
=======
def _greedy(boxes, container, order, deadline, pos_strategy, score_strategy):
    """Greedy placement with given position and scoring strategies, using grid for overlap checks."""
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    total_vol = 0.0
    n = len(boxes)
    
    # Use grid-based checks if container is reasonably sized
    use_grid = (W * H * D) <= 500000
    grid = None
    if use_grid:
        grid, _, _ = _build_grid(container, placed)
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        if pos_strategy == 'extreme':
            candidates = _extreme_points(placed, container)
            if (0, 0, 0) not in candidates:
                candidates.append((0, 0, 0))
        else:  # faces
            candidates = _face_points(placed, container)
        
        best_pos = None
        best_rot = None
        best_key = None
        
        for pos in candidates:
            for rw, rh, rd in rots:
                if use_grid:
                    fits = _fits_grid(grid, pos[0], pos[1], pos[2], rw, rh, rd)
                else:
                    fits = _fits((rw, rh, rd), pos, container, placed)
                
                if fits:
                    if score_strategy == 'compact':
                        key = _score_position_compact(pos, (rw, rh, rd), placed, container)
                    else:
                        key = _score_position_low(pos, (rw, rh, rd), placed, container)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = pos
                        best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_rot
            placed.append((x, y, z, rw, rh, rd))
            placements[idx] = (0, x, y, z, rw, rh, rd)
            total_vol += rw * rh * rd
            if use_grid:
                grid[int(x):int(x+rw), int(y):int(y+rh), int(z):int(z+rd)] = 1
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements, total_vol
>>>>>>> REPLACE
<<<<<<< SEARCH
def _local_search(placements, boxes, container, deadline, rng, max_iters=50):
    """Local search: remove and re-insert boxes to improve packing."""
    W, H, D = container
    n = len(boxes)
    
    placed = []
    for i in range(n):
        p = placements[i]
        if p[0] == 0:
            placed.append((p[1], p[2], p[3], p[4], p[5], p[6]))
    
    loaded_indices = [i for i in range(n) if placements[i][0] == 0]
    if not loaded_indices:
        return placements, 0.0
    
    improved = False
    for _ in range(min(max_iters, len(loaded_indices) * 2)):
        if time.time() > deadline:
            break
        idx = rng.choice(loaded_indices)
        p = placements[idx]
        box_entry = (p[1], p[2], p[3], p[4], p[5], p[6])
        
        # Remove
        try:
            placed.remove(box_entry)
        except ValueError:
            placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
            continue
        
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        # Try to place it back at a better position
        candidates = _extreme_points(placed, container)
        if (0, 0, 0) not in candidates:
            candidates.append((0, 0, 0))
        
        best_pos = None
        best_rot = None
        best_key = None
        for pos in candidates:
            for rw, rh, rd in rots:
                if _fits((rw, rh, rd), pos, container, placed):
                    key = _score_position_compact(pos, (rw, rh, rd), placed, container)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = pos
                        best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_rot
            placed.append((x, y, z, rw, rh, rd))
            placements[idx] = (0, x, y, z, rw, rh, rd)
            improved = True
        else:
            # Put it back
            placed.append(box_entry)
    
    total_vol = sum(placements[i][4] * placements[i][5] * placements[i][6] for i in range(n) if placements[i][0] == 0)
    return placements, total_vol
=======
def _local_search(placements, boxes, container, deadline, rng, max_iters=50):
    """Local search: remove and re-insert boxes to improve packing, using grid for overlap checks."""
    W, H, D = container
    n = len(boxes)
    
    placed = []
    for i in range(n):
        p = placements[i]
        if p[0] == 0:
            placed.append((p[1], p[2], p[3], p[4], p[5], p[6]))
    
    loaded_indices = [i for i in range(n) if placements[i][0] == 0]
    if not loaded_indices:
        return placements, 0.0
    
    # Use grid-based checks if container is reasonably sized
    use_grid = (W * H * D) <= 500000
    grid = None
    if use_grid:
        grid, _, _ = _build_grid(container, placed)
    
    improved = False
    for _ in range(min(max_iters, len(loaded_indices) * 2)):
        if time.time() > deadline:
            break
        idx = rng.choice(loaded_indices)
        p = placements[idx]
        box_entry = (p[1], p[2], p[3], p[4], p[5], p[6])
        
        # Remove from placed list
        try:
            placed.remove(box_entry)
        except ValueError:
            placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
            continue
        
        # Clear from grid
        if use_grid:
            px, py, pz, pw, ph, pd = box_entry
            grid[int(px):int(px+pw), int(py):int(py+ph), int(pz):int(pz+pd)] = 0
        
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        # Try to place it back at a better position
        candidates = _extreme_points(placed, container)
        if (0, 0, 0) not in candidates:
            candidates.append((0, 0, 0))
        
        best_pos = None
        best_rot = None
        best_key = None
        for pos in candidates:
            for rw, rh, rd in rots:
                if use_grid:
                    fits = _fits_grid(grid, pos[0], pos[1], pos[2], rw, rh, rd)
                else:
                    fits = _fits((rw, rh, rd), pos, container, placed)
                
                if fits:
                    key = _score_position_compact(pos, (rw, rh, rd), placed, container)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = pos
                        best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_rot
            placed.append((x, y, z, rw, rh, rd))
            placements[idx] = (0, x, y, z, rw, rh, rd)
            if use_grid:
                grid[int(x):int(x+rw), int(y):int(y+rh), int(z):int(z+rd)] = 1
            improved = True
        else:
            # Put it back
            placed.append(box_entry)
            if use_grid:
                px, py, pz, pw, ph, pd = box_entry
                grid[int(px):int(px+pw), int(py):int(py+ph), int(pz):int(pz+pd)] = 1
    
    total_vol = sum(placements[i][4] * placements[i][5] * placements[i][6] for i in range(n) if placements[i][0] == 0)
    return placements, total_vol
>>>>>>> REPLACE
BET: {mechanism: "Using NumPy grid-based overlap checks replaces O(n) list iteration in _fits with O(1) array slicing, enabling faster evaluation of candidate positions and allowing more greedy restarts and local search iterations within the 120s deadline", q: 0.65, delta: [0.0, 0.15]}
