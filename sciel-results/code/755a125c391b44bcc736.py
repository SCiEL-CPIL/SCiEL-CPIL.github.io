import time
import random
import itertools
from collections import defaultdict

def _rotations(w, h, d):
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

def _candidate_positions(placed, container, mode='extreme'):
    """Generate candidate positions for placing a new box."""
    W, H, D = container
    candidates = set()
    candidates.add((0, 0, 0))
    
    if mode == 'extreme':
        # Use face-based extreme points (3 faces per box for speed)
        for (px, py, pz, pw, ph, pd) in placed:
            # Right face
            if px + pw <= W:
                candidates.add((px + pw, py, pz))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
            # Front face
            if py + ph <= H:
                candidates.add((px, py + ph, pz))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
            # Top face
            if pz + pd <= D:
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py, pz + pd + 0))  # same
                candidates.add((px + pw, py + ph, pz + pd))
    
    return list(candidates)

def _position_score(pos, dim, container):
    """Score a position - lower is better (pack tighter)."""
    W, H, D = container
    x, y, z = pos
    # Prefer lower z, then lower y, then lower x
    return (z, y, x)

def _greedy_place(boxes, container, order, deadline, scoring_func=None, rng=None):
    """Place boxes greedily in the given order."""
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    
    if scoring_func is None:
        scoring_func = _position_score
    
    for orig_idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[orig_idx]
        rots = _rotations(w, h, d)
        candidates = _candidate_positions(placed, container, 'extreme')
        
        best_pos = None
        best_dim = None
        best_score = None
        
        for (cx, cy, cz) in candidates:
            for (rw, rh, rd) in rots:
                if not _fits((rw, rh, rd), (cx, cy, cz), container):
                    continue
                if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                    continue
                
                score = scoring_func((cx, cy, cz), (rw, rh, rd), container)
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_dim = (rw, rh, rd)
        
        if best_pos is not None:
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

def _get_placed_list(placements):
    placed = []
    for i, p in enumerate(placements):
        if p[0] == 0:
            placed.append((p[1], p[2], p[3], p[4], p[5], p[6]))
    return placed

def _try_insert_box(box_idx, boxes, placements, container, rng=None):
    """Try to insert a box at any valid position."""
    w, h, d = boxes[box_idx]
    rots = _rotations(w, h, d)
    placed = _get_placed_list(placements)
    candidates = _candidate_positions(placed, container, 'extreme')
    
    if rng:
        rng.shuffle(candidates)
    
    for (cx, cy, cz) in candidates:
        for (rw, rh, rd) in rots:
            if not _fits((rw, rh, rd), (cx, cy, cz), container):
                continue
            if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                continue
            return (0, cx, cy, cz, rw, rh, rd)
    return None

def _local_search(initial_placements, boxes, container, rng, deadline, max_rounds=3):
    """Local search: insert unplaced, relocate placed to improve geometry."""
    current_placements = [tuple(p) for p in initial_placements]
    current_volume = sum(p[4]*p[5]*p[6] for p in current_placements if p[0]==0)
    best_placements = [tuple(p) for p in current_placements]
    best_volume = current_volume
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        
        # Try to insert unplaced boxes
        unplaced = [i for i, p in enumerate(current_placements) if p[0] == -1]
        for idx in unplaced:
            if time.time() > deadline:
                break
            new_pl = _try_insert_box(idx, boxes, current_placements, container, rng)
            if new_pl:
                vol = new_pl[4]*new_pl[5]*new_pl[6]
                current_placements[idx] = new_pl
                current_volume += vol
                improved = True
        
        if current_volume > best_volume:
            best_volume = current_volume
            best_placements = [tuple(p) for p in current_placements]
        
        if not improved and round_num > 0:
            break
        
        # Try relocating placed boxes
        placed_idx = [i for i, p in enumerate(current_placements) if p[0] == 0]
        rng.shuffle(placed_idx)
        for idx in placed_idx:
            if time.time() > deadline:
                break
            
            old_pl = current_placements[idx]
            old_vol = old_pl[4]*old_pl[5]*old_pl[6]
            
            # Remove box
            current_placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
            current_volume -= old_vol
            
            # Try to re-insert at different position
            new_pl = _try_insert_box(idx, boxes, current_placements, container, rng)
            
            if new_pl:
                new_vol = new_pl[4]*new_pl[5]*new_pl[6]
                current_placements[idx] = new_pl
                current_volume += new_vol
                if current_volume > best_volume:
                    best_volume = current_volume
                    best_placements = [tuple(p) for p in current_placements]
            else:
                current_placements[idx] = old_pl
                current_volume += old_vol
    
    return best_placements, best_volume

def _shake_local_search(initial_placements, boxes, container, rng, deadline, max_rounds=2):
    """Shake: randomly remove some boxes, reinsert, then local search."""
    current_placements = [tuple(p) for p in initial_placements]
    current_volume = sum(p[4]*p[5]*p[6] for p in current_placements if p[0]==0)
    best_placements = [tuple(p) for p in current_placements]
    best_volume = current_volume
    
    placed_idx = [i for i, p in enumerate(current_placements) if p[0] == 0]
    if not placed_idx:
        return best_placements, best_volume
    
    # Remove 10-20% of placed boxes randomly
    n_remove = max(1, int(len(placed_idx) * 0.15))
    to_remove = rng.sample(placed_idx, min(n_remove, len(placed_idx)))
    for idx in to_remove:
        p = current_placements[idx]
        current_volume -= p[4]*p[5]*p[6]
        current_placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
    
    # Reinsert removed boxes
    for idx in to_remove:
        if time.time() > deadline:
            break
        new_pl = _try_insert_box(idx, boxes, current_placements, container, rng)
        if new_pl:
            current_placements[idx] = new_pl
            current_volume += new_pl[4]*new_pl[5]*new_pl[6]
    
    if current_volume > best_volume:
        best_volume = current_volume
        best_placements = [tuple(p) for p in current_placements]
    
    # Run local search on the shaken solution
    ls_placements, ls_volume = _local_search(current_placements, boxes, container, rng, deadline, max_rounds=1)
    if ls_volume > best_volume:
        best_volume = ls_volume
        best_placements = ls_placements
    
    return best_placements, best_volume

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    
    volumes = [b[0]*b[1]*b[2] for b in boxes]
    max_dims = [max(b) for b in boxes]
    min_dims = [min(b) for b in boxes]
    sum_dims = [sum(b) for b in boxes]
    surface_areas = [2*(b[0]*b[1] + b[1]*b[2] + b[0]*b[2]) for b in boxes]
    
    # Generate diverse initial solutions
    strategies = []
    
    vol_order = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    strategies.append((vol_order, None, None))
    
    maxdim_order = sorted(range(n), key=lambda i: max_dims[i], reverse=True)
    strategies.append((maxdim_order, None, None))
    
    mindim_order = sorted(range(n), key=lambda i: min_dims[i])
    strategies.append((mindim_order, None, None))
    
    sumdim_order = sorted(range(n), key=lambda i: sum_dims[i], reverse=True)
    strategies.append((sumdim_order, None, None))
    
    sa_order = sorted(range(n), key=lambda i: surface_areas[i], reverse=True)
    strategies.append((sa_order, None, None))
    
    mixed1 = sorted(range(n), key=lambda i: (volumes[i], max_dims[i]), reverse=True)
    strategies.append((mixed1, None, None))
    
    mixed2 = sorted(range(n), key=lambda i: (surface_areas[i], volumes[i]), reverse=True)
    strategies.append((mixed2, None, None))
    
    # Custom scoring functions
    def score_sum(pos, dim, cont):
        return pos[0] + pos[1] + pos[2]
        
    def score_max(pos, dim, cont):
        return max(pos)
        
    def score_z_priority(pos, dim, cont):
        return (pos[2], pos[0] + pos[1])
        
    def score_compact(pos, dim, cont):
        # Prefer positions that minimize max extent
        return max(pos[0] + dim[0], pos[1] + dim[1], pos[2] + dim[2])
    
    strategies.append((vol_order, score_sum, None))
    strategies.append((vol_order, score_max, None))
    strategies.append((vol_order, score_z_priority, None))
    strategies.append((vol_order, score_compact, None))
    
    # Random orders
    rng_main = random.Random(42)
    for _ in range(8):
        order = list(range(n))
        rng_main.shuffle(order)
        strategies.append((order, None, random.Random(rng_main.randint(1, 1000))))
    
    best_placements = None
    best_volume = -1
    
    for order, scorer, rng in strategies:
        if time.time() > deadline - 50:
            break
        placements, vol = _greedy_place(boxes, container, order, deadline - 50, scorer, rng)
        
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            if best_volume >= container_volume:
                break
    
    if best_placements is None:
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
        best_placements = placements
        best_volume = sum(p[4]*p[5]*p[6] for p in placements if p[0]==0)

    # Phase 2: Local search with multiple seeds
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 5:
            break
        rng_ls = random.Random(seed)
        ls_placements, ls_volume = _local_search(best_placements, boxes, container, rng_ls, deadline, max_rounds=2)
        
        if ls_volume > best_volume:
            best_volume = ls_volume
            best_placements = ls_placements
            if best_volume >= container_volume:
                break
    
    # Phase 3: Shake local search for escaping local optima
    for seed in [10, 11, 12, 13, 14, 15]:
        if time.time() > deadline - 5:
            break
        rng_shake = random.Random(seed)
        shake_placements, shake_volume = _shake_local_search(best_placements, boxes, container, rng_shake, deadline, max_rounds=1)
        
        if shake_volume > best_volume:
            best_volume = shake_volume
            best_placements = shake_placements
            if best_volume >= container_volume:
                break
    
    return best_placements
