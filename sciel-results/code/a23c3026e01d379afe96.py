import time
import random
from itertools import permutations
import numpy as np

def load(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # Check if we can use a voxel grid
    # Assume integer dimensions for voxel grid. If not, we might need to scale or use list.
    # For now, let's assume integers as is common in these tasks.
    # If dimensions are floats, we can't easily use a simple int grid without scaling.
    # Let's check types.
    use_voxel = False
    if all(isinstance(d, int) for d in container):
        # Estimate memory. 100x100x100 = 1M bytes. 200x200x200 = 8M bytes.
        # 500x500x500 = 125M bytes. Might be tight but okay for int8.
        if W * H * D < 50_000_000:
            use_voxel = True

    def get_rotations(w, h, d):
        seen = set()
        rots = []
        for p in permutations((w, h, d)):
            if p not in seen:
                seen.add(p)
                rots.append(p)
        return rots

    # Pre-compute rotations
    box_rots = []
    for b in boxes:
        box_rots.append(get_rotations(*b))

    # Voxel Grid Implementation
    if use_voxel:
        def solve_voxel(box_indices, sort_key, seed=None):
            if seed is not None:
                random.seed(seed)
            
            # Create grid
            grid = np.zeros((W, H, D), dtype=np.int8)
            
            items = []
            for i in box_indices:
                w, h, d = boxes[i]
                items.append((i, w, h, d))
            
            # Sort items
            items.sort(key=lambda x: sort_key(x[1], x[2], x[3]), reverse=True)
            
            # To handle ties or add randomness, we can shuffle items with similar keys?
            # Or just shuffle the candidate order.
            
            candidates = [(0, 0, 0)]
            placements = [None] * len(boxes)
            
            for i, w, h, d in items:
                best_placement = None
                rots = box_rots[i][:]
                if seed is not None:
                    random.shuffle(rots)
                
                # Sort candidates by BLF (Bottom, Left, Front) -> (z, y, x)
                # Using z as height, y as depth, x as width?
                # Container is (W, H, D).
                # Usually x is width, y is height, z is depth?
                # The output format is (bin, x, y, z, w, h, d).
                # Let's assume x in [0,W], y in [0,H], z in [0,D].
                # "Bottom" usually means lowest z? Or lowest y?
                # In 3D packing, usually one axis is vertical.
                # Let's stick to the previous logic: sort by (z, y, x) where z is the "height" axis?
                # The previous code sorted by (p[2], p[1], p[0]) which is (z, y, x).
                # And bounds were cx+rw > W, cy+rh > H, cz+rd > D.
                # So x->W, y->H, z->D.
                # So "Bottom" is low z? No, usually z is up.
                # If z is up, we want low z.
                # If y is up, we want low y.
                # The previous code sorted by z first. Let's assume z is the vertical axis.
                
                sorted_candidates = sorted(candidates, key=lambda p: (p[2], p[1], p[0]))
                
                found = False
                for cx, cy, cz in sorted_candidates:
                    if found:
                        break
                    for rw, rh, rd in rots:
                        # Check bounds
                        if cx + rw > W or cy + rh > H or cz + rd > D:
                            continue
                        
                        # Check overlap using vectorized slice
                        # grid[x, y, z]
                        # slice is grid[cx:cx+rw, cy:cy+rh, cz:cz+rd]
                        # Note: NumPy slicing is [start:stop], so stop is exclusive.
                        if grid[cx:cx+rw, cy:cy+rh, cz:cz+rd].any():
                            continue
                        
                        # If we get here, it fits
                        best_placement = (cx, cy, cz, rw, rh, rd)
                        found = True
                        break
                
                if best_placement:
                    x, y, z, rw, rh, rd = best_placement
                    # Mark as occupied
                    grid[x:x+rw, y:y+rh, z:z+rd] = 1
                    placements[i] = (0, x, y, z, rw, rh, rd)
                    
                    # Generate new candidates
                    new_pts = [
                        (x + rw, y, z),
                        (x, y + rh, z),
                        (x, y, z + rd),
                        (x + rw, y + rh, z),
                        (x + rw, y, z + rd),
                        (x, y + rh, z + rd),
                        (x + rw, y + rh, z + rd)
                    ]
                    for p in new_pts:
                        if 0 <= p[0] <= W and 0 <= p[1] <= H and 0 <= p[2] <= D:
                            candidates.append(p)
                else:
                    placements[i] = (-1, 0, 0, 0, w, h, d)
            
            return placements

    else:
        # Fallback: List-based
        def solve_list(box_indices, sort_key, seed=None):
            if seed is not None:
                random.seed(seed)
            
            placed = []
            items = []
            for i in box_indices:
                w, h, d = boxes[i]
                items.append((i, w, h, d))
            
            items.sort(key=lambda x: sort_key(x[1], x[2], x[3]), reverse=True)
            
            candidates = [(0, 0, 0)]
            placements = [None] * len(boxes)
            
            for i, w, h, d in items:
                best_placement = None
                rots = box_rots[i][:]
                if seed is not None:
                    random.shuffle(rots)
                
                sorted_candidates = sorted(candidates, key=lambda p: (p[2], p[1], p[0]))
                
                found = False
                for cx, cy, cz in sorted_candidates:
                    if found:
                        break
                    for rw, rh, rd in rots:
                        if cx + rw > W or cy + rh > H or cz + rd > D:
                            continue
                        
                        # Check overlap
                        overlap = False
                        for px, py, pz, pw, ph, pd in placed:
                            if not (cx + rw <= px or px + pw <= cx or
                                    cy + rh <= py or py + ph <= cy or
                                    cz + rd <= pz or pz + pd <= cz):
                                overlap = True
                                break
                        if overlap:
                            continue
                        
                        best_placement = (cx, cy, cz, rw, rh, rd)
                        found = True
                        break
                
                if best_placement:
                    x, y, z, rw, rh, rd = best_placement
                    placed.append((x, y, z, rw, rh, rd))
                    placements[i] = (0, x, y, z, rw, rh, rd)
                    
                    new_pts = [
                        (x + rw, y, z),
                        (x, y + rh, z),
                        (x, y, z + rd),
                        (x + rw, y + rh, z),
                        (x + rw, y, z + rd),
                        (x, y + rh, z + rd),
                        (x + rw, y + rh, z + rd)
                    ]
                    for p in new_pts:
                        if 0 <= p[0] <= W and 0 <= p[1] <= H and 0 <= p[2] <= D:
                            candidates.append(p)
                else:
                    placements[i] = (-1, 0, 0, 0, w, h, d)
            
            return placements

    def score_placements(placements):
        total_vol = 0
        for p in placements:
            if p[0] == 0:
                total_vol += p[4] * p[5] * p[6]
        return total_vol

    indices = list(range(len(boxes)))
    best_score = -1
    best_placements = None

    # Sort keys
    sort_vol = lambda w, h, d: w * h * d
    sort_max = lambda w, h, d: max(w, h, d)
    sort_sum = lambda w, h, d: w + h + d
    sort_area = lambda w, h, d: 2 * (w * h + h * d + w * d)
    
    strategies = [
        sort_vol,
        sort_max,
        sort_sum,
        sort_area
    ]

    solver = solve_voxel if use_voxel else solve_list

    # Run deterministic strategies first
    for key in strategies:
        if time.time() > deadline:
            break
        pl = solver(indices, key)
        sc = score_placements(pl)
        if sc > best_score:
            best_score = sc
            best_placements = pl

    # Run randomized restarts
    # With voxel grid, each run is very fast.
    # We can run many more.
    max_restarts = 200 if use_voxel else 20
    
    for seed in range(max_restarts):
        if time.time() > deadline:
            break
        # Randomly pick a strategy
        key = random.choice(strategies)
        pl = solver(indices, key, seed)
        sc = score_placements(pl)
        if sc > best_score:
            best_score = sc
            best_placements = pl

    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_placements
