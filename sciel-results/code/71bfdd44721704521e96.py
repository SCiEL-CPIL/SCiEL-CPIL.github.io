"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from collections import defaultdict
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]
    
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill
    
    container_vol = W * H * D
    
    # Precompute unique orientations for each box
    orient = []
    for w, h, d in boxes:
        opts = set()
        for p in itertools.permutations((w, h, d)):
            opts.add(p)
        orient.append(list(opts))
    
    box_vol = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    
    # Check if container dimensions are reasonable for bitmap approach
    # If container is too large, bitmap would use too much memory
    max_bitmap_size = 200 * 200 * 200  # 8 million cells
    use_bitmap = (W * H * D <= max_bitmap_size) and all(
        isinstance(x, int) and x > 0 for x in (W, H, D)
    ) and all(
        all(isinstance(v, int) and v > 0 for v in box) for box in boxes
    )
    
    if use_bitmap:
        # Create occupancy bitmap
        occupancy = np.zeros((W, H, D), dtype=np.uint8)
        
        def try_pack_bitmap(order, time_limit):
            """
            Fast 3D pack using bitmap-based overlap checking.
            Returns (filled_volume, placements_list)
            """
            placed = []  # list of (x, y, z, w, h, d, box_idx)
            filled = 0
            
            # Candidate positions: maintain a set of potential (x, y, z) positions
            # Start with (0, 0, 0)
            candidates = [(0, 0, 0)]
            candidate_set = {(0, 0, 0)}
            
            for idx in order:
                w, h, d = boxes[idx]
                best_pos = None
                best_key = None
                
                # Try all orientations
                for (ow, oh, od) in orient[idx]:
                    if ow > W or oh > H or od > D:
                        continue
                    
                    # Try all candidate positions
                    for (cx, cy, cz) in candidates:
                        if cx + ow > W or cy + oh > H or cz + od > D:
                            continue
                        
                        # Check overlap using bitmap (vectorized)
                        region = occupancy[cx:cx+ow, cy:cy+oh, cz:cz+od]
                        if np.any(region > 0):
                            continue
                        
                        # Valid position
                        key = (cz, cy, cx)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_pos = (cx, cy, cz, ow, oh, od)
                        
                        # Early exit if we found the best possible position
                        if best_key == (0, 0, 0):
                            break
                    
                    if best_key == (0, 0, 0):
                        break
                
                if best_pos is not None:
                    bx, by, bz, bw, bh, bd = best_pos
                    placed.append((bx, by, bz, bw, bh, bd, idx))
                    filled += bw * bh * bd
                    
                    # Mark occupancy
                    occupancy[bx:bx+bw, by:by+bh, bz:bz+bd] = 1
                    
                    # Generate new candidate positions
                    new_candidates = []
                    if bz + bd <= D:
                        new_candidates.append((bx, by, bz + bd))
                    if bx + bw <= W:
                        new_candidates.append((bx + bw, by, bz))
                    if by + bh <= H:
                        new_candidates.append((bx, by + bh, bz))
                    if bx + bw <= W and by + bh <= H:
                        new_candidates.append((bx + bw, by + bh, bz))
                    if bx + bw <= W and bz + bd <= D:
                        new_candidates.append((bx + bw, by, bz + bd))
                    if by + bh <= H and bz + bd <= D:
                        new_candidates.append((bx, by + bh, bz + bd))
                    if bx + bw <= W and by + bh <= H and bz + bd <= D:
                        new_candidates.append((bx + bw, by + bh, bz + bd))
                    
                    for nc in new_candidates:
                        if nc not in candidate_set:
                            candidate_set.add(nc)
                            candidates.append(nc)
                    
                    # Time check
                    if time.time() > time_limit:
                        break
            
            return filled, placed
        
        def try_pack(order, time_limit, orient_mode=0):
            """Wrapper that uses bitmap if available"""
            return try_pack_bitmap(order, time_limit)
    else:
        # Fallback to original list-based approach
        def try_pack(order, time_limit, orient_mode=0):
            """
            Fast 3D pack using anchor-based placement.
            Returns (filled_volume, placements_list)
            """
            placed = []  # list of (x, y, z, w, h, d, box_idx)
            filled = 0
            
            # Anchors: (x, y, z)
            anchors = [(0, 0, 0)]
            anchor_set = {(0, 0, 0)}
            
            for idx in order:
                w, h, d = boxes[idx]
                best_pos = None
                best_key = None
                
                for (ow, oh, od) in orient[idx]:
                    if ow > W or oh > H or od > D:
                        continue
                    
                    for (ax, ay, az) in anchors:
                        if ax + ow > W or ay + oh > H or az + od > D:
                            continue
                        
                        # Overlap check
                        ok = True
                        for (px, py, pz, pw, ph, pd, pidx) in placed:
                            if not (ax + ow <= px or px + pw <= ax or
                                    ay + oh <= py or py + ph <= ay or
                                    az + od <= pz or pz + pd <= az):
                                ok = False
                                break
                        
                        if ok:
                            key = (az, ay, ax)
                            if best_key is None or key < best_key:
                                best_key = key
                                best_pos = (ax, ay, az, ow, oh, od)
                            
                            if best_key == (0, 0, 0):
                                break
                    
                    if best_key == (0, 0, 0):
                        break
                
                if best_pos is not None:
                    bx, by, bz, bw, bh, bd = best_pos
                    placed.append((bx, by, bz, bw, bh, bd, idx))
                    filled += bw * bh * bd
                    
                    new_anchors = []
                    if bz + bd <= D:
                        new_anchors.append((bx, by, bz + bd))
                    if bx + bw <= W:
                        new_anchors.append((bx + bw, by, bz))
                    if by + bh <= H:
                        new_anchors.append((bx, by + bh, bz))
                    if bx + bw <= W and by + bh <= H:
                        new_anchors.append((bx + bw, by + bh, bz))
                    if bx + bw <= W and bz + bd <= D:
                        new_anchors.append((bx + bw, by, bz + bd))
                    if by + bh <= H and bz + bd <= D:
                        new_anchors.append((bx, by + bh, bz + bd))
                    if bx + bw <= W and by + bh <= H and bz + bd <= D:
                        new_anchors.append((bx + bw, by + bh, bz + bd))
                    
                    for na in new_anchors:
                        if na not in anchor_set:
                            anchor_set.add(na)
                            anchors.append(na)
                    
                    if time.time() > time_limit:
                        break
            
            return filled, placed
    
    def make_orders():
        orders = []
        orders.append(sorted(range(n), key=lambda i: -box_vol[i]))
        orders.append(sorted(range(n), key=lambda i: box_vol[i]))
        orders.append(sorted(range(n), key=lambda i: -min(boxes[i])))
        orders.append(sorted(range(n), key=lambda i: -max(boxes[i])))
        def surf(i):
            a, b, c = boxes[i]
            return 2 * (a * b + a * c + b * c)
        orders.append(sorted(range(n), key=lambda i: -surf(i)))
        def compact(i):
            a, b, c = boxes[i]
            m = max(a, b, c)
            return (a * b * c) / m if m else 0
        orders.append(sorted(range(n), key=lambda i: -compact(i)))
        def balance(i):
            a, b, c = sorted(boxes[i])
            if c == 0:
                return 0
            return (a * b) / c
        orders.append(sorted(range(n), key=lambda i: -balance(i)))
        orders.append(sorted(range(n), key=lambda i: -(sum(boxes[i]))))
        return orders
    
    best_filled = -1
    best_placed = []
    best_order = None
    
    base_orders = make_orders()
    
    # Run base orders
    for order in base_orders:
        if time.time() > deadline:
            break
        filled, placed = try_pack(order, deadline)
        if filled > best_filled:
            best_filled = filled
            best_placed = placed
            best_order = list(order)
    
    # Iterated Local Search (ILS)
    rng = random.Random(12345)
    attempts = 0
    max_attempts = 5000  # Increased attempts due to faster bitmap checks
    
    while time.time() < deadline and attempts < max_attempts:
        if best_order is None:
            order = list(rng.choice(base_orders))
        else:
            order = list(best_order)
            
        strategy = rng.random()
        if strategy < 0.3:
            for _ in range(rng.randint(1, 5)):
                i, j = rng.randrange(n), rng.randrange(n)
                order[i], order[j] = order[j], order[i]
        elif strategy < 0.6:
            seg_len = rng.randint(2, min(15, n))
            start = rng.randrange(max(1, n - seg_len))
            order[start:start+seg_len] = reversed(order[start:start+seg_len])
        elif strategy < 0.8:
            i = rng.randrange(n)
            elem = order.pop(i)
            j = rng.randrange(len(order) + 1)
            order.insert(j, elem)
        else:
            rng.shuffle(order)
        
        filled, placed = try_pack(order, deadline)
        if filled > best_filled:
            best_filled = filled
            best_placed = placed
            best_order = list(order)
        attempts += 1
        
        if best_filled >= container_vol:
            break
    
    # Build result
    result = [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]
    for (px, py, pz, pw, ph, pd, pidx) in best_placed:
        result[pidx] = (0, px, py, pz, pw, ph, pd)
    
    return result
# EVOLVE-BLOCK-END
