import time
import random
import itertools

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
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    
    if scoring_func is None:
        def scoring_func(pos, dim):
            return (pos[2], pos[1], pos[0])
    
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

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 148
    
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    
    volumes = [b[0]*b[1]*b[2] for b in boxes]
    max_dims = [max(b) for b in boxes]
    min_dims = [min(b) for b in boxes]
    sum_dims = [sum(b) for b in boxes]
    surface_areas = [2*(b[0]*b[1] + b[1]*b[2] + b[0]*b[2]) for b in boxes]
    
    # Phase 1: Generate diverse initial solutions
    strategies = []
    
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
    
    def score_sum(pos, dim):
        return pos[0] + pos[1] + pos[2]
        
    def score_max(pos, dim):
        return max(pos)
        
    def score_z_sum(pos, dim):
        return (pos[2], pos[0] + pos[1])
        
    strategies.append(('eap', vol_order, score_sum, None))
    strategies.append(('eap', vol_order, score_max, None))
    strategies.append(('eap', vol_order, score_z_sum, None))
    
    rngs = [random.Random(i) for i in range(10)]
    for r in rngs:
        strategies.append(('eap', vol_order, None, r))
        
    rng_main = random.Random(42)
    for _ in range(15):
        order = list(range(n))
        rng_main.shuffle(order)
        strategies.append(('eap', order, None, random.Random(rng_main.randint(1, 1000))))

    best_placements = None
    best_volume = -1
    
    for algo, order, scorer, rng in strategies:
        if time.time() > deadline - 40:
            break
        placements, vol = _try_place_eap(boxes, container, order, deadline - 40, scorer, rng)
        
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            if best_volume >= container_volume:
                break
    
    if best_placements is None:
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

    # Phase 2: Multi-start Local Search with Box Relocation and Swap
    # We'll run multiple local search rounds starting from the best initial solution
    
    def get_placed_list(placements):
        placed = []
        for i, p in enumerate(placements):
            if p[0] == 0:
                placed.append((p[1], p[2], p[3], p[4], p[5], p[6]))
        return placed
    
    def try_insert_box(box_idx, placements, container, rng=None, shuffle_candidates=False):
        w, h, d = boxes[box_idx]
        rots = _rotations(w, h, d)
        placed = get_placed_list(placements)
        candidates = _extreme_points(placed, container)
        
        if rng and shuffle_candidates and len(candidates) > 1:
            rng.shuffle(candidates)
        
        for (cx, cy, cz) in candidates:
            for (rw, rh, rd) in rots:
                if not _fits((rw, rh, rd), (cx, cy, cz), container):
                    continue
                if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                    continue
                return (0, cx, cy, cz, rw, rh, rd)
        return None
    
    # Local search procedure
    def local_search(initial_placements, rng, deadline, max_rounds=3):
        current_placements = [tuple(p) for p in initial_placements]
        current_volume = sum(p[4]*p[5]*p[6] for p in current_placements if p[0]==0)
        ls_best_placements = [tuple(p) for p in current_placements]
        ls_best_volume = current_volume
        
        for round_num in range(max_rounds):
            improved = False
            
            # Try to insert unplaced boxes
            unplaced = [i for i, p in enumerate(current_placements) if p[0] == -1]
            for idx in unplaced:
                if time.time() > deadline:
                    break
                new_pl = try_insert_box(idx, current_placements, container, rng)
                if new_pl:
                    vol = new_pl[4]*new_pl[5]*new_pl[6]
                    current_placements[idx] = new_pl
                    current_volume += vol
                    improved = True
            
            if current_volume > ls_best_volume:
                ls_best_volume = current_volume
                ls_best_placements = [tuple(p) for p in current_placements]
            
            if not improved:
                break
            
            # Try relocating placed boxes to create better geometry
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
                new_pl = try_insert_box(idx, current_placements, container, rng, shuffle_candidates=True)
                
                if new_pl:
                    new_vol = new_pl[4]*new_pl[5]*new_pl[6]
                    current_placements[idx] = new_pl
                    current_volume += new_vol
                    # Volume is same, but geometry changed - accept to explore
                    
                    if current_volume > ls_best_volume:
                        ls_best_volume = current_volume
                        ls_best_placements = [tuple(p) for p in current_placements]
                else:
                    # Failed to re-insert, revert
                    current_placements[idx] = old_pl
                    current_volume += old_vol
            
            if time.time() > deadline:
                break
        
        return ls_best_placements, ls_best_volume
    
    # Run multiple local searches with different random seeds
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 5:
            break
        rng_ls = random.Random(seed)
        ls_placements, ls_volume = local_search(best_placements, rng_ls, deadline, max_rounds=2)
        
        if ls_volume > best_volume:
            best_volume = ls_volume
            best_placements = ls_placements
            if best_volume >= container_volume:
                break
    
    return best_placements
