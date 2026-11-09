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

def _grid_check(grid, x, y, z, w, h, d):
    """Check if a box fits in the grid at position (x,y,z) with dims (w,h,d)."""
    # Check bounds first to avoid slicing errors
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w >= grid.shape[0] or y + h >= grid.shape[1] or z + d >= grid.shape[2]:
        return False
    # Check if any cell in the sub-volume is occupied (1)
    return not np.any(grid[x:x+w, y:y+h, z:z+d] == 1)

def _grid_place(grid, x, y, z, w, h, d, val=1):
    grid[x:x+w, y:y+h, z:z+d] = val

def load(boxes, container):
    start_time = time.time()
    # 150s limit, keep margin
    deadline = start_time + 140
    
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    
    # 1. Initial Solution via EAP
    # We use a few standard orders to pick the best starting point
    volumes = [b[0]*b[1]*b[2] for b in boxes]
    vol_order = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    
    best_initial_placements, best_initial_vol = _try_place_eap(boxes, container, vol_order, deadline)
    
    # 2. Simulated Annealing
    # Setup Grid
    # Grid resolution: We use integer coordinates. If W, H, D are large, 
    # a full grid might be memory heavy. 
    # Given typical container sizes in these tasks (e.g. 10x10x10 to 100x100x100),
    # a 100^3 grid is 1,000,000 floats/integers, which is ~8MB. This is safe.
    # If dimensions are huge, we might need to scale, but let's assume standard scale.
    # To be safe against memory blowup for very large containers, we cap grid size?
    # No, the task says "60-box instances". Boxes are likely integer dimensions.
    
    # Initialize Grid with best initial solution
    grid = np.zeros((W, H, D), dtype=np.int8)
    current_placements = best_initial_placements[:]
    current_vol = best_initial_vol
    
    # Populate grid
    for p in current_placements:
        if p[0] == 0:
            _, x, y, z, w, h, d = p
            _grid_place(grid, x, y, z, w, h, d, 1)
    
    # SA Parameters
    T = 100.0  # Initial temperature
    cooling = 0.995
    min_T = 0.1
    rng = random.Random(42)
    
    # Pre-compute rotations for each box to speed up moves
    box_rots = [_rotations(b[0], b[1], b[2]) for b in boxes]
    
    # Lists for indices
    placed_indices = [i for i, p in enumerate(current_placements) if p[0] == 0]
    unplaced_indices = [i for i, p in enumerate(current_placements) if p[0] == -1]
    
    # Helper to update grid and placement list
    def remove_box(i):
        if current_placements[i][0] == 0:
            _, x, y, z, w, h, d = current_placements[i]
            _grid_place(grid, x, y, z, w, h, d, 0)
            # Remove from placed list
            if i in placed_indices:
                placed_indices.remove(i)
            unplaced_indices.append(i)
            current_placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
            
    def add_box(i, x, y, z, w, h, d):
        current_placements[i] = (0, x, y, z, w, h, d)
        _grid_place(grid, x, y, z, w, h, d, 1)
        if i in unplaced_indices:
            unplaced_indices.remove(i)
        placed_indices.append(i)

    best_placements = current_placements[:]
    best_volume = current_vol
    
    while time.time() < deadline and T > min_T:
        # Pick a random move type
        # 1. Move a placed box to a new location
        # 2. Swap a placed box with an unplaced box (effectively replacing)
        # 3. Rotate a placed box in place (if it fits)
        
        move_type = rng.random()
        
        if move_type < 0.5 and placed_indices:
            # Move placed box
            i = rng.choice(placed_indices)
            old_p = current_placements[i]
            old_x, old_y, old_z = old_p[1], old_p[2], old_p[3]
            old_w, old_h, old_d = old_p[4], old_p[5], old_p[6]
            
            # Remove
            _grid_place(grid, old_x, old_y, old_z, old_w, old_h, old_d, 0)
            placed_indices.remove(i)
            
            # Try to place at random location
            # Sample random position
            success = False
            for _ in range(10): # Try a few times
                rw, rh, rd = rng.choice(box_rots[i])
                # Random position within bounds
                rx = rng.randint(0, max(0, W - rw))
                ry = rng.randint(0, max(0, H - rh))
                rz = rng.randint(0, max(0, D - rd))
                
                if _grid_check(grid, rx, ry, rz, rw, rh, rd):
                    _grid_place(grid, rx, ry, rz, rw, rh, rd, 1)
                    current_placements[i] = (0, rx, ry, rz, rw, rh, rd)
                    placed_indices.append(i)
                    success = True
                    break
            
            if not success:
                # Could not place, so it's now unplaced
                unplaced_indices.append(i)
                current_placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
            
            # Calculate delta volume
            # Old volume was always added, new volume is added if success
            # Actually, we need to compare total volume
            # Easier: just recalculate? No, too slow.
            # Delta = New_Vol - Old_Vol
            # Old_Vol contributed old_w*old_h*old_d
            # New_Vol contributes rw*rh*rd if success, else 0
            # But wait, if we moved it, the space it occupied is freed.
            # The "score" is total volume filled.
            # So if we move a box of volume V to a new spot, total volume doesn't change.
            # Unless it fails to place, then total volume decreases by V.
            
            if success:
                delta = 0
            else:
                delta = - (boxes[i][0]*boxes[i][1]*boxes[i][2])
                
        elif move_type < 0.8 and unplaced_indices and placed_indices:
            # Swap unplaced with placed? Or just try to place unplaced?
            # Let's try to place an unplaced box into the grid.
            i = rng.choice(unplaced_indices)
            # Remove from unplaced list temporarily? No, we check fit first.
            
            success = False
            new_vol = 0
            for _ in range(10):
                rw, rh, rd = rng.choice(box_rots[i])
                rx = rng.randint(0, max(0, W - rw))
                ry = rng.randint(0, max(0, H - rh))
                rz = rng.randint(0, max(0, D - rd))
                
                if _grid_check(grid, rx, ry, rz, rw, rh, rd):
                    _grid_place(grid, rx, ry, rz, rw, rh, rd, 1)
                    current_placements[i] = (0, rx, ry, rz, rw, rh, rd)
                    unplaced_indices.remove(i)
                    placed_indices.append(i)
                    success = True
                    new_vol = rw * rh * rd
                    break
            
            if success:
                delta = new_vol
            else:
                delta = 0 # No change
                
        else:
            # Random restart or perturbation?
            # Let's just cool down
            T *= cooling
            continue

        # Accept/Reject
        new_vol = current_vol + delta
        if new_vol > current_vol:
            current_vol = new_vol
            if current_vol > best_volume:
                best_volume = current_vol
                best_placements = current_placements[:]
        else:
            # Probability of accepting worse solution
            if T > 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_vol = new_vol
                    # Accept
                else:
                    # Reject: Revert
                    # We need to revert the specific move
                    # This is tricky because we mutated the state.
                    # For simplicity and robustness in this implementation,
                    # we will only accept improvements or keep state if we can't easily revert.
                    # Given the complexity of reverting specific grid changes in a generic way,
                    # and the fact that "move" often results in 0 delta or negative delta,
                    # let's simplify: Only accept if delta >= 0.
                    # This makes it a Hill Climber, not SA.
                    # To make it SA, we need to snapshot.
                    # Snapshotting the grid (1M ints) is ~8MB copy. 
                    # Doing this every step is too slow.
                    
                    # Alternative: Only try moves that are likely good or keep a small history?
                    # Let's stick to Hill Climbing with random restarts if we get stuck.
                    # Or, we can just not revert and let the "bad" move persist if we accept?
                    # No, if we reject, we MUST revert.
                    
                    # Let's implement a simple revert for the specific cases.
                    if move_type < 0.5:
                        # Move placed box
                        if success:
                            # Revert move: remove new, restore old
                            _, rx, ry, rz, rw, rh, rd = current_placements[i]
                            _grid_place(grid, rx, ry, rz, rw, rh, rd, 0)
                            placed_indices.remove(i)
                            _grid_place(grid, old_x, old_y, old_z, old_w, old_h, old_d, 1)
                            placed_indices.append(i)
                            current_placements[i] = (0, old_x, old_y, old_z, old_w, old_h, old_d)
                        else:
                            # It was unplaced. Revert: it was placed.
                            unplaced_indices.remove(i)
                            _grid_place(grid, old_x, old_y, old_z, old_w, old_h, old_d, 1)
                            placed_indices.append(i)
                            current_placements[i] = (0, old_x, old_y, old_z, old_w, old_h, old_d)
                    elif move_type < 0.8:
                        # Place unplaced box
                        if success:
                            # Revert: remove new
                            _, rx, ry, rz, rw, rh, rd = current_placements[i]
                            _grid_place(grid, rx, ry, rz, rw, rh, rd, 0)
                            placed_indices.remove(i)
                            unplaced_indices.append(i)
                            current_placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
                        # else: no change to revert

        T *= cooling

    return best_placements
