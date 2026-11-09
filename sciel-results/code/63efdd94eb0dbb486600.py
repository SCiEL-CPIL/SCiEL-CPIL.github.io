"""3D Container Loading: Combined skyline + shelf + multi-strategy search with improved scoring."""
import time
import random
import itertools

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box (w,h,d)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _fits(box_dims, pos, container, placed):
    """Check if box fits at pos without overlap and within container."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if x >= px + pw or x + w <= px:
            continue
        if y >= py + ph or y + h <= py:
            continue
        if z >= pz + pd or z + d <= pz:
            continue
        return False
    return True

def _candidate_positions(placed, container, mode='extreme'):
    """Generate candidate (x,y,z) positions from placed boxes."""
    W, H, D = container
    
    if not placed:
        return [(0, 0, 0)]
    
    if mode == 'extreme':
        xs = {0}
        ys = {0}
        zs = {0}
        for (px, py, pz, pw, ph, pd) in placed:
            xs.add(px + pw)
            ys.add(py + ph)
            zs.add(pz + pd)
        xs = sorted(x for x in xs if x < W)
        ys = sorted(y for y in ys if y < H)
        zs = sorted(z for z in zs if z < D)
        points = []
        for x in xs:
            for y in ys:
                for z in zs:
                    points.append((x, y, z))
        return points
    elif mode == 'faces':
        candidates = set()
        candidates.add((0, 0, 0))
        for (px, py, pz, pw, ph, pd) in placed:
            for dx in (0, pw):
                for dy in (0, ph):
                    for dz in (0, pd):
                        x, y, z = px + dx, py + dy, pz + dz
                        if x < W and y < H and z < D:
                            candidates.add((x, y, z))
        return list(candidates)
    else:  # combined
        xs = {0}
        ys = {0}
        zs = {0}
        for (px, py, pz, pw, ph, pd) in placed:
            xs.add(px + pw)
            ys.add(py + ph)
            zs.add(pz + pd)
        xs = sorted(x for x in xs if x < W)
        ys = sorted(y for y in ys if y < H)
        zs = sorted(z for z in zs if z < D)
        points = set()
        for x in xs:
            for y in ys:
                for z in zs:
                    points.add((x, y, z))
        # Also add face positions
        for (px, py, pz, pw, ph, pd) in placed:
            for dx in (0, pw):
                for dy in (0, ph):
                    for dz in (0, pd):
                        x, y, z = px + dx, py + dy, pz + dz
                        if x < W and y < H and z < D:
                            points.add((x, y, z))
        return list(points)

def _score_position(pos, box_dims, strategy='compact'):
    """Score a candidate position. Lower is better."""
    x, y, z = pos
    w, h, d = box_dims
    if strategy == 'compact':
        # Prefer positions that fill the bottom-left-front more completely
        return (z + d, y + h, x + w)
    elif strategy == 'low':
        return (z, y, x)
    elif strategy == 'back':
        return (-(x + w), -(y + h), -(z + d))
    elif strategy == 'corner':
        return (-(x + w + y + h + z + d),)
    else:
        return (z + d, y + h, x + w)

def _greedy_place(boxes, container, order, deadline, mode='extreme', strategy='compact'):
    """Greedy placement with given order and mode."""
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
        
        candidates = _candidate_positions(placed, container, mode)
        
        best_pos = None
        best_rot = None
        best_key = None
        
        for pos in candidates:
            for rw, rh, rd in rots:
                if _fits((rw, rh, rd), pos, container, placed):
                    key = _score_position(pos, (rw, rh, rd), strategy)
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

def _shelf_place(boxes, container, order, deadline):
    """Shelf-based packing: partition container into horizontal shelves."""
    W, H, D = container
    n = len(boxes)
    placed = []
    placements = [None] * n
    total_vol = 0.0
    
    # Shelf state: list of (shelf_z, shelf_height, list of (x, w) occupied on this shelf)
    shelves = []
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        best_pos = None
        best_rot = None
        best_key = None
        
        for rw, rh, rd in rots:
            # Try to fit on existing shelves
            for si, (sz, sh, occupied) in enumerate(shelves):
                if sz + rh > H:
                    continue
                # Check if box fits on this shelf (in x-z plane)
                # Shelf occupies z from sz to sz+sh, x from 0 to W, y from 0 to D
                # Actually shelf is a horizontal layer: z fixed range, x and y vary
                # Let me reconsider: shelf at height z with height rh
                # Box needs to fit in x [0,W] and y [0,D] at z level
                # Check against occupied boxes on this shelf
                
                can_fit = True
                for (ox, oy, oz, ow, oh, od) in placed:
                    if oz >= sz and oz + od <= sz + rh:
                        # Same shelf
                        if not (ox + ow <= x or x + rw <= ox):
                            pass
                        # Need to check x-y overlap at this z level
                        # Actually this is getting complex. Let me simplify.
                        pass
                
                # Simplified: just check if box fits in container at this shelf level
                # and doesn't overlap with placed boxes
                # Try placing at x=0, y=0 on this shelf
                test_pos = (0, 0, sz)
                if _fits((rw, rh, rd), test_pos, container, placed):
                    key = _score_position(test_pos, (rw, rh, rd), 'compact')
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = test_pos
                        best_rot = (rw, rh, rd)
                else:
                    # Try shifting x
                    for ox in range(0, W - rw + 1, max(1, rw // 4)):
                        test_pos = (ox, 0, sz)
                        if _fits((rw, rh, rd), test_pos, container, placed):
                            key = _score_position(test_pos, (rw, rh, rd), 'compact')
                            if best_key is None or key < best_key:
                                best_key = key
                                best_pos = test_pos
                                best_rot = (rw, rh, rd)
                            break
            
            # Try new shelf at top
            new_shelf_z = max((p[2] + p[5] for p in placed), default=0)
            if new_shelf_z + rh <= H:
                test_pos = (0, 0, new_shelf_z)
                if _fits((rw, rh, rd), test_pos, container, placed):
                    key = _score_position(test_pos, (rw, rh, rd), 'compact')
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = test_pos
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

def _local_search(placements, boxes, container, deadline, rng, max_iters=80):
    """Local search: remove and re-place boxes to improve packing."""
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
    
    total_vol = sum(boxes[i][0] * boxes[i][1] * boxes[i][2] for i in loaded_indices)
    
    improved = True
    iters = 0
    while improved and iters < max_iters and time.time() < deadline:
        improved = False
        iters += 1
        
        for idx in loaded_indices:
            if time.time() > deadline:
                break
            
            p = placements[idx]
            box_pos = (p[1], p[2], p[3], p[4], p[5], p[6])
            
            # Try removing this box
            try:
                placed.remove(box_pos)
            except ValueError:
                continue
            
            w, h, d = boxes[idx]
            rots = _rotations(w, h, d)
            candidates = _candidate_positions(placed, container, 'extreme')
            
            best_pos = None
            best_rot = None
            best_key = None
            
            for pos in candidates:
                for rw, rh, rd in rots:
                    if _fits((rw, rh, rd), pos, container, placed):
                        key = _score_position(pos, (rw, rh, rd), 'compact')
                        if best_key is None or key < best_key:
                            best_key = key
                            best_pos = pos
                            best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                x, y, z = best_pos
                rw, rh, rd = best_rot
                new_pos = (x, y, z, rw, rh, rd)
                
                # Check if this is an improvement
                old_score = _score_position(box_pos[:3], box_pos[3:], 'compact')
                new_score = _score_position((x, y, z), (rw, rh, rd), 'compact')
                
                if new_score < old_score:
                    placed.append(new_pos)
                    placements[idx] = (0, x, y, z, rw, rh, rd)
                    improved = True
                else:
                    placed.append(box_pos)
            else:
                placed.append(box_pos)
    
    total_vol = 0.0
    for i in range(n):
        if placements[i][0] == 0:
            total_vol += placements[i][4] * placements[i][5] * placements[i][6]
    
    return placements, total_vol

def _shake_local_search(placements, boxes, container, deadline, rng, max_shakes=30):
    """Shake local search: randomly remove a few boxes and re-place them."""
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
    
    total_vol = sum(boxes[i][0] * boxes[i][1] * boxes[i][2] for i in loaded_indices)
    
    best_placements = [p[:] for p in placements]
    best_vol = total_vol
    
    for shake in range(max_shakes):
        if time.time() > deadline:
            break
        
        num_remove = min(rng.randint(1, 3), len(loaded_indices))
        to_remove = rng.sample(loaded_indices, num_remove)
        
        removed_boxes = []
        for idx in to_remove:
            p = placements[idx]
            box_pos = (p[1], p[2], p[3], p[4], p[5], p[6])
            try:
                placed.remove(box_pos)
            except ValueError:
                continue
            removed_boxes.append((idx, box_pos))
        
        success = True
        new_placements = [p[:] for p in placements]
        
        for idx, old_pos in removed_boxes:
            w, h, d = boxes[idx]
            rots = _rotations(w, h, d)
            candidates = _candidate_positions(placed, container, 'extreme')
            
            best_pos = None
            best_rot = None
            best_key = None
            
            for pos in candidates:
                for rw, rh, rd in rots:
                    if _fits((rw, rh, rd), pos, container, placed):
                        key = _score_position(pos, (rw, rh, rd), 'compact')
                        if best_key is None or key < best_key:
                            best_key = key
                            best_pos = pos
                            best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                x, y, z = best_pos
                rw, rh, rd = best_rot
                placed.append((x, y, z, rw, rh, rd))
                new_placements[idx] = (0, x, y, z, rw, rh, rd)
            else:
                success = False
                break
        
        if success:
            new_vol = 0.0
            for i in range(n):
                if new_placements[i][0] == 0:
                    new_vol += new_placements[i][4] * new_placements[i][5] * new_placements[i][6]
            
            if new_vol > best_vol:
                best_vol = new_vol
                best_placements = [p[:] for p in new_placements]
                placed = []
                for i in range(n):
                    p = best_placements[i]
                    if p[0] == 0:
                        placed.append((p[1], p[2], p[3], p[4], p[5], p[6]))
        else:
            for idx, old_pos in removed_boxes:
                try:
                    placed.remove(old_pos)
                except ValueError:
                    pass
                placed.append(old_pos)
    
    return best_placements, best_vol

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 135.0
    W, H, D = container
    n = len(boxes)
    
    vols = [w * h * d for w, h, d in boxes]
    mins = [min(w, h, d) for w, h, d in boxes]
    maxs = [max(w, h, d) for w, h, d in boxes]
    ratios = [max(w, h, d) / max(min(w, h, d), 1e-9) for w, h, d in boxes]
    
    best_placements = None
    best_vol = 0.0
    
    sort_strategies = [
        ('vol_desc', lambda i: vols[i], True),
        ('vol_asc', lambda i: vols[i], False),
        ('min_dim_desc', lambda i: mins[i], True),
        ('min_dim_asc', lambda i: mins[i], False),
        ('max_dim_desc', lambda i: maxs[i], True),
        ('max_dim_asc', lambda i: maxs[i], False),
        ('ratio_desc', lambda i: ratios[i], True),
        ('ratio_asc', lambda i: ratios[i], False),
    ]
    
    modes = ['extreme', 'faces', 'combined']
    strategies = ['compact', 'low']
    
    for sort_name, key_fn, reverse in sort_strategies:
        if time.time() > deadline:
            break
        
        order = list(range(n))
        order.sort(key=key_fn, reverse=reverse)
        
        for mode in modes:
            if time.time() > deadline:
                break
            
            for strat in strategies:
                if time.time() > deadline:
                    break
                
                placements, vol = _greedy_place(boxes, container, order, deadline, mode, strat)
                if vol > best_vol:
                    best_vol = vol
                    best_placements = [p[:] for p in placements]
        
        # Shelf-based packing
        if time.time() < deadline:
            placements, vol = _shelf_place(boxes, container, order, deadline)
            if vol > best_vol:
                best_vol = vol
                best_placements = [p[:] for p in placements]
    
    # Local search on best
    if best_placements is not None and time.time() < deadline:
        rng = random.Random(42)
        best_placements, best_vol = _local_search(best_placements, boxes, container, deadline, rng, max_iters=80)
    
    # Random restarts
    rng = random.Random(123)
    base_order = list(range(n))
    base_order.sort(key=lambda i: vols[i], reverse=True)
    
    remaining_time = deadline - time.time()
    num_restarts = int(min(200, max(20, remaining_time / 2.0)))
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        order = base_order[:]
        num_swaps = rng.randint(2, min(12, n))
        for _ in range(num_swaps):
            a, b = rng.randrange(n), rng.randrange(n)
            if a != b:
                order[a], order[b] = order[b], order[a]
        
        mode = rng.choice(modes)
        strat = rng.choice(strategies)
        
        placements, vol = _greedy_place(boxes, container, order, deadline, mode, strat)
        if vol > best_vol:
            best_vol = vol
            best_placements = [p[:] for p in placements]
        
        # Also try shelf
        if time.time() < deadline:
            placements, vol = _shelf_place(boxes, container, order, deadline)
            if vol > best_vol:
                best_vol = vol
                best_placements = [p[:] for p in placements]
    
    # Final shake local search
    if best_placements is not None and time.time() < deadline:
        rng2 = random.Random(999)
        best_placements, best_vol = _shake_local_search(best_placements, boxes, container, deadline, rng2, max_shakes=40)
    
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        order = list(range(n))
        order.sort(key=lambda i: vols[i], reverse=True)
        placements, vol = _greedy_place(boxes, container, order, deadline, 'extreme', 'compact')
        if vol > 0:
            best_placements = placements
    
    return best_placements
