import time
import random
import itertools
import sys

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
    # Unpack for speed
    for px, py, pz, pw, ph, pd in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _extreme_points(placed, container):
    W, H, D = container
    candidates = set()
    candidates.add((0, 0, 0))
    for px, py, pz, pw, ph, pd in placed:
        # Generate points from faces
        # x-face
        candidates.add((px + pw, py, pz))
        candidates.add((px + pw, py + ph, pz))
        candidates.add((px + pw, py, pz + pd))
        candidates.add((px + pw, py + ph, pz + pd))
        
        # y-face
        candidates.add((px, py + ph, pz))
        candidates.add((px + pw, py + ph, pz))
        candidates.add((px, py + ph, pz + pd))
        candidates.add((px + pw, py + ph, pz + pd))
        
        # z-face
        candidates.add((px, py, pz + pd))
        candidates.add((px + pw, py, pz + pd))
        candidates.add((px, py + ph, pz + pd))
        candidates.add((px + pw, py + ph, pz + pd))
        
    # Filter valid points
    valid = []
    for cx, cy, cz in candidates:
        if 0 <= cx <= W and 0 <= cy <= H and 0 <= cz <= D:
            valid.append((cx, cy, cz))
    return valid

def _greedy_load(boxes, container, order, rng=None, scoring=None):
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    
    # Pre-compute rotations for all boxes
    box_rots = [_rotations(b[0], b[1], b[2]) for b in boxes]
    
    for idx in order:
        w, h, d = boxes[idx]
        rots = box_rots[idx]
        
        candidates = _extreme_points(placed, container)
        
        best_pos = None
        best_dim = None
        best_score = None
        best_candidates = []
        
        for cx, cy, cz in candidates:
            for rw, rh, rd in rots:
                if not _fits((rw, rh, rd), (cx, cy, cz), container):
                    continue
                if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                    continue
                
                # Scoring: prefer lower z, then lower y, then lower x
                # This fills from bottom-up, left-to-right, front-to-back
                if scoring:
                    score = scoring((cx, cy, cz), (rw, rh, rd))
                else:
                    score = (cz, cy, cx)
                
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
            placements[idx] = (0, best_pos[0], best_pos[1], best_pos[2],
                               best_dim[0], best_dim[1], best_dim[2])
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
            
    # Fill any unprocessed (should not happen if order is permutation)
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    volume = sum(p[4]*p[5]*p[6] for p in placements if p[0] == 0)
    return placements, volume

def _local_search(boxes, container, placements, deadline, rng):
    W, H, D = container
    n = len(boxes)
    current_placements = [tuple(p) for p in placements]
    current_volume = sum(p[4]*p[5]*p[6] for p in current_placements if p[0] == 0)
    best_placements = [tuple(p) for p in current_placements]
    best_volume = current_volume
    
    box_rots = [_rotations(b[0], b[1], b[2]) for b in boxes]
    
    def get_placed_list(placements):
        return [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in placements if p[0] == 0]
    
    def try_insert(box_idx, placed_list):
        w, h, d = boxes[box_idx]
        rots = box_rots[box_idx]
        candidates = _extreme_points(placed_list, container)
        
        # Shuffle candidates to avoid bias
        if rng:
            rng.shuffle(candidates)
            
        for cx, cy, cz in candidates:
            for rw, rh, rd in rots:
                if not _fits((rw, rh, rd), (cx, cy, cz), container):
                    continue
                if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed_list):
                    continue
                return (0, cx, cy, cz, rw, rh, rd)
        return None

    while time.time() < deadline:
        placed_indices = [i for i, p in enumerate(current_placements) if p[0] == 0]
        unplaced_indices = [i for i, p in enumerate(current_placements) if p[0] == -1]
        
        if not unplaced_indices:
            break
            
        # Try to insert a random unplaced box
        target_idx = rng.choice(unplaced_indices)
        placed_list = get_placed_list(current_placements)
        new_pos = try_insert(target_idx, placed_list)
        
        if new_pos:
            # Success: volume increases
            old_vol = current_placements[target_idx][4]*current_placements[target_idx][5]*current_placements[target_idx][6] # 0
            current_placements[target_idx] = new_pos
            current_volume += new_pos[4]*new_pos[5]*new_pos[6]
            
            if current_volume > best_volume:
                best_volume = current_volume
                best_placements = [tuple(p) for p in current_placements]
        else:
            # Failed: try to move a placed box to make room
            if not placed_indices:
                continue
                
            # Pick a random placed box to move
            mover_idx = rng.choice(placed_indices)
            old_mover_pos = current_placements[mover_idx]
            
            # Remove mover
            current_placements[mover_idx] = (-1, 0, 0, 0, boxes[mover_idx][0], boxes[mover_idx][1], boxes[mover_idx][2])
            placed_list = get_placed_list(current_placements)
            
            # Try to insert target
            new_target_pos = try_insert(target_idx, placed_list)
            
            if new_target_pos:
                # Target fits now. Try to re-insert mover
                current_placements[target_idx] = new_target_pos
                current_volume += new_target_pos[4]*new_target_pos[5]*new_target_pos[6]
                
                # Try to re-insert mover
                placed_list = get_placed_list(current_placements)
                new_mover_pos = try_insert(mover_idx, placed_list)
                
                if new_mover_pos:
                    # Both fit! Volume is same as before (target added, mover moved)
                    # Wait, target was unplaced, now placed. Mover was placed, now placed.
                    # Net change: +Volume(Target)
                    current_placements[mover_idx] = new_mover_pos
                    # Mover volume is same, so current_volume is correct (added target)
                    
                    if current_volume > best_volume:
                        best_volume = current_volume
                        best_placements = [tuple(p) for p in current_placements]
                else:
                    # Mover doesn't fit back. Revert.
                    current_placements[target_idx] = (-1, 0, 0, 0, boxes[target_idx][0], boxes[target_idx][1], boxes[target_idx][2])
                    current_placements[mover_idx] = old_mover_pos
                    current_volume -= new_target_pos[4]*new_target_pos[5]*new_target_pos[6]
            else:
                # Target still doesn't fit. Revert mover.
                current_placements[mover_idx] = old_mover_pos

    return best_placements, best_volume

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 145 # Safety margin
    
    W, H, D = container
    n = len(boxes)
    
    volumes = [b[0]*b[1]*b[2] for b in boxes]
    
    # Strategy 1: Multiple Greedy Runs
    best_placements = None
    best_volume = -1
    
    # Define different sorting strategies
    strategies = []
    vol_order = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    strategies.append(vol_order)
    
    maxdim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    strategies.append(maxdim_order)
    
    mindim_order = sorted(range(n), key=lambda i: min(boxes[i]))
    strategies.append(mindim_order)
    
    sumdim_order = sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True)
    strategies.append(sumdim_order)
    
    # Run deterministic greeds
    for order in strategies:
        if time.time() > deadline - 60: break
        placements, vol = _greedy_load(boxes, container, order, rng=None)
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            
    # Run randomized greeds
    rng_master = random.Random(42)
    num_random_runs = 50
    for i in range(num_random_runs):
        if time.time() > deadline - 60: break
        
        # Random order
        order = list(range(n))
        rng_master.shuffle(order)
        
        # Random tie-breaking
        rng = random.Random(i)
        
        placements, vol = _greedy_load(boxes, container, order, rng=rng)
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            
    if best_placements is None:
        # Fallback
        best_placements = [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]
        best_volume = 0
        
    # Strategy 2: Local Search on the best result
    if time.time() < deadline:
        rng_ls = random.Random(123)
        best_placements, best_volume = _local_search(boxes, container, best_placements, deadline, rng_ls)
        
    return best_placements
