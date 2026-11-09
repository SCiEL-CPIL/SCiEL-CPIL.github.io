import time
import random
import itertools
import numpy as np

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits(box_dims, pos, container):
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    return x + w <= W and y + h <= H and z + d <= D

def _no_overlap(box_dims, pos, placed):
    x, y, z = pos
    w, h, d = box_dims
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _extreme_points(placed, container):
    W, H, D = container
    candidates = set()
    candidates.add((0, 0, 0))
    for (px, py, pz, pw, ph, pd) in placed:
        points = [
            (px + pw, py, pz),
            (px, py + ph, pz),
            (px, py, pz + pd),
            (px + pw, py + ph, pz),
            (px, py + ph, pz + pd),
            (px + pw, py, pz + pd),
            (px + pw, py + ph, pz + pd),
        ]
        for (cx, cy, cz) in points:
            if cx <= W and cy <= H and cz <= D:
                candidates.add((cx, cy, cz))
    return list(candidates)

def _try_place_eap(boxes, container, order_idx, deadline, scoring_func=None, rng=None):
    """Extreme Points Approach with optional custom scoring and randomization."""
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    
    if scoring_func is None:
        def scoring_func(pos, dim):
            return (pos[2], pos[1], pos[0]) # Bottom, Left, Far
    
    for orig_idx in order_idx:
        if time.time() > deadline:
            break
        w, h, d = boxes[orig_idx]
        rots = _rotations(w, h, d)
        candidates = _extreme_points(placed, container)
        
        best_pos = None
        best_dim = None
        best_score = None
        best_candidates = []
        
        for (cx, cy, cz) in candidates:
            for (rw, rh, rd) in rots:
                if not _fits((rw, rh, rd), (cx, cy, cz), container):
                    continue
                if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                    continue
                
                score = scoring_func((cx, cy, cz), (rw, rh, rd))
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_dim = (rw, rh, rd)
                    best_candidates = [(best_pos, best_dim)]
                elif score == best_score:
                    best_candidates.append(((cx, cy, cz), (rw, rh, rd)))
        
        if best_pos is not None:
            if rng and len(best_candidates) > 1:
                best_pos, best_dim = rng.choice(best_candidates)
            
            placed.append((best_pos[0], best_pos[1], best_pos[2],
                          best_dim[0], best_dim[1], best_dim[2]))
            placements[orig_idx] = (0, best_pos[0], best_pos[1], best_pos[2],
                                   best_dim[0], best_dim[1], best_dim[2])
        else:
            placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    volume_filled = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
    return placements, volume_filled

def _try_place_grid(boxes, container, order_idx, deadline, scoring_func=None, rng=None, grid_resolution=20):
    """
    Greedy placement using a 3D occupancy grid for fast overlap checks.
    """
    W, H, D = container
    n = len(boxes)
    placements = [None] * n
    
    # Grid dimensions
    gw = max(1, int(grid_resolution))
    gh = max(1, int(grid_resolution))
    gd = max(1, int(grid_resolution))
    
    # Grid cell sizes
    cw = W / gw
    ch = H / gh
    cd = D / gd
    
    # Initialize grid as 0 (free)
    grid = np.zeros((gw, gh, gd), dtype=np.uint8)
    
    if scoring_func is None:
        def scoring_func(pos, dim):
            return (pos[2], pos[1], pos[0])
    
    for orig_idx in order_idx:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[orig_idx]
        rots = _rotations(w, h, d)
        candidates = _extreme_points(
            [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in placements if p is not None and p[0] == 0],
            container
        )
        
        best_pos = None
        best_dim = None
        best_score = None
        best_candidates = []
        
        for (cx, cy, cz) in candidates:
            for (rw, rh, rd) in rots:
                if not _fits((rw, rh, rd), (cx, cy, cz), container):
                    continue
                
                # Check overlap using grid
                # Convert physical coords to grid indices
                gx0 = int(cx / cw)
                gy0 = int(cy / ch)
                gz0 = int(cz / cd)
                
                gx1 = int((cx + rw - 1e-9) / cw)
                gy1 = int((cy + rh - 1e-9) / ch)
                gz1 = int((cz + rd - 1e-9) / cd)
                
                # Clamp to grid bounds
                gx0 = max(0, min(gx0, gw - 1))
                gy0 = max(0, min(gy0, gh - 1))
                gz0 = max(0, min(gz0, gd - 1))
                gx1 = max(0, min(gx1, gw - 1))
                gy1 = max(0, min(gy1, gh - 1))
                gz1 = max(0, min(gz1, gd - 1))
                
                # Check if any cell in the range is occupied
                if grid[gx0:gx1+1, gy0:gy1+1, gz0:gz1+1].any():
                    continue
                
                # Also do a precise overlap check to avoid grid approximation errors
                # This is a fallback for safety, but the grid check is the main speedup
                if not _no_overlap((rw, rh, rd), (cx, cy, cz), 
                                   [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in placements if p is not None and p[0] == 0]):
                    continue
                
                score = scoring_func((cx, cy, cz), (rw, rh, rd))
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_dim = (rw, rh, rd)
                    best_candidates = [(best_pos, best_dim)]
                elif score == best_score:
                    best_candidates.append(((cx, cy, cz), (rw, rh, rd)))
        
        if best_pos is not None:
            if rng and len(best_candidates) > 1:
                best_pos, best_dim = rng.choice(best_candidates)
            
            x, y, z = best_pos
            rw, rh, rd = best_dim
            
            # Update grid
            gx0 = int(x / cw)
            gy0 = int(y / ch)
            gz0 = int(z / cd)
            gx1 = int((x + rw - 1e-9) / cw)
            gy1 = int((y + rh - 1e-9) / ch)
            gz1 = int((z + rd - 1e-9) / cd)
            
            gx0 = max(0, min(gx0, gw - 1))
            gy0 = max(0, min(gy0, gh - 1))
            gz0 = max(0, min(gz0, gd - 1))
            gx1 = max(0, min(gx1, gw - 1))
            gy1 = max(0, min(gy1, gh - 1))
            gz1 = max(0, min(gz1, gd - 1))
            
            grid[gx0:gx1+1, gy0:gy1+1, gz0:gz1+1] = 1
            
            placements[orig_idx] = (0, x, y, z, rw, rh, rd)
        else:
            placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    volume_filled = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
    return placements, volume_filled

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 140
    
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    
    volumes = [b[0]*b[1]*b[2] for b in boxes]
    max_dims = [max(b) for b in boxes]
    min_dims = [min(b) for b in boxes]
    sum_dims = [sum(b) for b in boxes]
    surface_areas = [2*(b[0]*b[1] + b[1]*b[2] + b[0]*b[2]) for b in boxes]
    
    strategies = []
    
    # Standard Heuristics (EAP)
    vol_order = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    strategies.append(('eap', vol_order, None, None))
    
    maxdim_order = sorted(range(n), key=lambda i: max_dims[i], reverse=True)
    strategies.append(('eap', maxdim_order, None, None))
    
    mindim_order = sorted(range(n), key=lambda i: min_dims[i])
    strategies.append(('eap', mindim_order, None, None))
    
    sumdim_order = sorted(range(n), key=lambda i: sum_dims[i], reverse=True)
    strategies.append(('eap', sumdim_order, None, None))
    
    sa_order = sorted(range(n), key=lambda i: surface_areas[i], reverse=True)
    strategies.append(('eap', sa_order, None, None))
    
    mixed1 = sorted(range(n), key=lambda i: (volumes[i], max_dims[i]), reverse=True)
    strategies.append(('eap', mixed1, None, None))
    
    # Custom Scoring Functions
    def score_sum(pos, dim):
        return pos[0] + pos[1] + pos[2]
        
    def score_max(pos, dim):
        return max(pos)
        
    def score_z_sum(pos, dim):
        return (pos[2], pos[0] + pos[1])
        
    strategies.append(('eap', vol_order, score_sum, None))
    strategies.append(('eap', vol_order, score_max, None))
    strategies.append(('eap', vol_order, score_z_sum, None))
    
    # Randomized Ties with Standard Score
    rngs = [random.Random(i) for i in range(10)]
    for r in rngs:
        strategies.append(('eap', vol_order, None, r))
        
    # Random Orders
    rng_main = random.Random(42)
    for _ in range(10):
        order = list(range(n))
        rng_main.shuffle(order)
        strategies.append(('eap', order, None, random.Random(rng_main.randint(1, 1000))))
    
    # Grid-based strategies
    # Use grid for some runs to see if it helps
    for i in range(5):
        rng = random.Random(100 + i)
        strategies.append(('grid', vol_order, None, rng))
        
    for i in range(5):
        rng = random.Random(200 + i)
        strategies.append(('grid', maxdim_order, None, rng))
    
    # Grid with random orders
    rng_main2 = random.Random(43)
    for _ in range(5):
        order = list(range(n))
        rng_main2.shuffle(order)
        strategies.append(('grid', order, None, random.Random(rng_main2.randint(1, 1000))))
        
    best_placements = None
    best_volume = -1
    
    for algo, order, scorer, rng in strategies:
        if time.time() > deadline:
            break
            
        if algo == 'eap':
            placements, vol = _try_place_eap(boxes, container, order, deadline, scorer, rng)
        elif algo == 'grid':
            placements, vol = _try_place_grid(boxes, container, order, deadline, scorer, rng)
        else:
            continue
            
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            if best_volume >= container_volume:
                break
    
    if best_placements is None:
        # Fallback
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
        best_placements = placements
    
    return best_placements
