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

def _extreme_points(placed, container):
    """Generate candidate (x,y,z) positions from placed boxes."""
    W, H, D = container
    xs = {0}
    ys = {0}
    zs = {0}
    for (px, py, pz, pw, ph, pd) in placed:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)
    
    # Filter to those within container bounds
    xs = [x for x in xs if x < W]
    ys = [y for y in ys if y < H]
    zs = [z for z in zs if z < D]
    
    points = []
    for x in xs:
        for y in ys:
            for z in zs:
                points.append((x, y, z))
    return points

def _fits(box_dims, pos, container, placed):
    """Check if box fits at pos without overlap and within container."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    
    # Check container bounds
    if x + w > W or y + h > H or z + d > D:
        return False
    
    # Check overlap with placed boxes
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _score_position(pos, box_dims, placed, container):
    """
    Score a candidate position.
    We want to minimize z (height), then y, then x to keep the packing compact.
    A lower score is better.
    """
    x, y, z = pos
    # Primary: keep it low (z)
    # Secondary: keep it forward (y)
    # Tertiary: keep it left (x)
    return (z, y, x)

def _greedy_place(boxes, container, order, deadline):
    """
    Greedy placement of boxes in the given order.
    Returns a list of placements (bin, x, y, z, w, h, d) and total volume.
    """
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
        
        # Generate candidate positions
        candidates = _extreme_points(placed, container)
        # Always include (0,0,0) just in case
        if (0, 0, 0) not in candidates:
            candidates.append((0, 0, 0))
        
        best_pos = None
        best_rot = None
        best_score = None
        
        for pos in candidates:
            for rw, rh, rd in rots:
                if _fits((rw, rh, rd), pos, container, placed):
                    score = _score_position(pos, (rw, rh, rd), placed, container)
                    if best_score is None or score < best_score:
                        best_score = score
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
    
    # Fill any unprocessed boxes as not placed
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements, total_vol

def _local_search(placements, boxes, container, deadline, rng):
    """
    Improve placement using shake and re-insertion.
    """
    n = len(boxes)
    current_placements = [p[:] for p in placements]
    current_vol = sum(p[4]*p[5]*p[6] for p in current_placements if p[0] == 0)
    
    loaded_indices = [i for i in range(n) if current_placements[i][0] == 0]
    
    if not loaded_indices:
        return current_placements, current_vol
    
    while time.time() < deadline:
        # Pick a random subset of loaded boxes to remove
        # Try varying sizes: small shakes for fine-tuning, larger for restructuring
        if rng.random() < 0.2 and len(loaded_indices) > 5:
            k = rng.randint(5, min(15, len(loaded_indices)))
        else:
            k = rng.randint(1, min(3, len(loaded_indices)))
            
        boxes_to_remove = rng.sample(loaded_indices, k)
        
        # Create a temporary state
        temp_placements = [p[:] for p in current_placements]
        
        # Extract placed boxes not in remove set
        placed_boxes = []
        for i in range(n):
            p = temp_placements[i]
            if p[0] == 0 and i not in boxes_to_remove:
                placed_boxes.append((p[1], p[2], p[3], p[4], p[5], p[6]))
        
        # Mark removed boxes as unplaced in temp
        for i in boxes_to_remove:
            w, h, d = boxes[i]
            temp_placements[i] = (-1, 0, 0, 0, w, h, d)
        
        # Try to re-insert removed boxes
        # Sort by volume descending to place big ones first? Or random?
        # Random order helps escape local minima.
        shuffle_list = boxes_to_remove[:]
        rng.shuffle(shuffle_list)
        
        for idx in shuffle_list:
            w, h, d = boxes[idx]
            rots = _rotations(w, h, d)
            candidates = _extreme_points(placed_boxes, container)
            if (0, 0, 0) not in candidates:
                candidates.append((0, 0, 0))
            
            best_pos = None
            best_rot = None
            best_score = None
            
            for pos in candidates:
                for rw, rh, rd in rots:
                    if _fits((rw, rh, rd), pos, container, placed_boxes):
                        score = _score_position(pos, (rw, rh, rd), placed_boxes, container)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = pos
                            best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                x, y, z = best_pos
                rw, rh, rd = best_rot
                placed_boxes.append((x, y, z, rw, rh, rd))
                temp_placements[idx] = (0, x, y, z, rw, rh, rd)
        
        # Calculate new volume
        new_vol = sum(p[4]*p[5]*p[6] for p in temp_placements if p[0] == 0)
        
        if new_vol > current_vol:
            current_placements = temp_placements
            current_vol = new_vol
            loaded_indices = [i for i in range(n) if current_placements[i][0] == 0]
            
            # Occasionally, if we made a big jump, do a full re-shuffle of all placed boxes
            # to see if we can pack them even better with a different order.
            if len(loaded_indices) > 10 and rng.random() < 0.1:
                # Extract all currently placed boxes
                all_placed = []
                for i in loaded_indices:
                    p = current_placements[i]
                    all_placed.append((i, (p[1], p[2], p[3], p[4], p[5], p[6])))
                
                # Shuffle the order of these boxes
                rng.shuffle(all_placed)
                
                # Re-greedy only these boxes
                # We need to clear the container and re-place these specific boxes
                temp_placements2 = [p[:] for p in current_placements]
                # Mark all as unplaced first? No, just the ones we are re-packing.
                # Actually, it's safer to re-pack ONLY these boxes into an empty container?
                # No, that would lose the context of other boxes if we had any not in this set.
                # But loaded_indices is ALL loaded boxes. So we are re-packing the entire set.
                
                # Clear all placements
                for i in range(n):
                    w, h, d = boxes[i]
                    temp_placements2[i] = (-1, 0, 0, 0, w, h, d)
                
                placed_boxes2 = []
                for idx, _ in all_placed:
                    w, h, d = boxes[idx]
                    rots = _rotations(w, h, d)
                    candidates = _extreme_points(placed_boxes2, container)
                    if (0, 0, 0) not in candidates:
                        candidates.append((0, 0, 0))
                    
                    best_pos = None
                    best_rot = None
                    best_score = None
                    
                    for pos in candidates:
                        for rw, rh, rd in rots:
                            if _fits((rw, rh, rd), pos, container, placed_boxes2):
                                score = _score_position(pos, (rw, rh, rd), placed_boxes2, container)
                                if best_score is None or score < best_score:
                                    best_score = score
                                    best_pos = pos
                                    best_rot = (rw, rh, rd)
                    
                    if best_pos is not None:
                        x, y, z = best_pos
                        rw, rh, rd = best_rot
                        placed_boxes2.append((x, y, z, rw, rh, rd))
                        temp_placements2[idx] = (0, x, y, z, rw, rh, rd)
                
                new_vol2 = sum(p[4]*p[5]*p[6] for p in temp_placements2 if p[0] == 0)
                if new_vol2 > current_vol:
                    current_placements = temp_placements2
                    current_vol = new_vol2
                    loaded_indices = [i for i in range(n) if current_placements[i][0] == 0]

    return current_placements, current_vol

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    W, H, D = container
    n = len(boxes)
    
    vols = [w * h * d for w, h, d in boxes]
    
    # Define sorting strategies
    sort_strategies = [
        ('vol_desc', lambda i: vols[i], True),
        ('min_dim_desc', lambda i: min(boxes[i]), True),
        ('max_dim_desc', lambda i: max(boxes[i]), True),
        ('ratio', lambda i: max(boxes[i]) / max(min(boxes[i]), 1e-9), False),
        ('vol_asc', lambda i: vols[i], False),
    ]
    
    best_placements = None
    best_vol = 0.0
    
    # 1. Run greedy with different sorts
    for name, key_fn, reverse in sort_strategies:
        if time.time() > deadline:
            break
        order = list(range(n))
        order.sort(key=key_fn, reverse=reverse)
        
        placements, vol = _greedy_place(boxes, container, order, deadline)
        if vol > best_vol:
            best_vol = vol
            best_placements = [p[:] for p in placements]
    
    if best_placements is None:
        # Fallback
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        return best_placements
    
    # 2. Local Search (Shake)
    rng = random.Random(42)
    best_placements, best_vol = _local_search(
        best_placements, boxes, container, deadline, rng
    )
    
    # 3. If time remains, try another greedy + local search with a different seed/sort
    if time.time() < deadline - 20:
        # Try a random permutation
        rng2 = random.Random(123)
        order = list(range(n))
        rng2.shuffle(order)
        
        placements, vol = _greedy_place(boxes, container, order, deadline)
        if vol > best_vol:
            best_vol = vol
            best_placements = [p[:] for p in placements]
        
        # Local search on this new best
        if time.time() < deadline - 5:
            best_placements, best_vol = _local_search(
                best_placements, boxes, container, deadline, rng2
            )
            
    return best_placements
