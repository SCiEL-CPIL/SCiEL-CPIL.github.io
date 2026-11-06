"""3D Container Loading - Anchor-based greedy with ILS and multiple heuristics."""
import time
import itertools
import random
import numpy as np

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
    
    # Create occupancy bitmap for fast overlap checking
    # Use float64 to avoid integer overflow issues with large dimensions
    occ = np.zeros((W + 1, H + 1, D + 1), dtype=np.int8)
    
    # Precompute unique orientations for each box
    orient = []
    for w, h, d in boxes:
        opts = set()
        for p in itertools.permutations((w, h, d)):
            opts.add(p)
        orient.append(list(opts))
    
    box_vol = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    
    def can_place_bitmap(x, y, z, w, h, d):
        """Check if a box fits at (x,y,z) with size (w,h,d) using occupancy bitmap."""
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check if the region is free
        region = occ[x:x+w, y:y+h, z:z+d]
        return np.all(region == 0)
    
    def place_box_bitmap(x, y, z, w, h, d):
        """Mark a box as placed in the occupancy bitmap."""
        occ[x:x+w, y:y+h, z:z+d] = 1
    
    def remove_box_bitmap(x, y, z, w, h, d):
        """Unmark a box in the occupancy bitmap."""
        occ[x:x+w, y:y+h, z:z+d] = 0
    
    def try_pack(order, time_limit, orient_mode=0):
        """
        Fast 3D pack using anchor-based placement with bitmap overlap checking.
        Returns (filled_volume, placements_list)
        """
        placed = []  # list of (x, y, z, w, h, d, box_idx)
        filled = 0
        
        # Anchors: (x, y, z)
        anchors = [(0, 0, 0)]
        anchor_set = {(0, 0, 0)}
        
        for idx in order:
            if time.time() > time_limit:
                break
                
            w, h, d = boxes[idx]
            best_pos = None
            best_key = None
            
            for (ow, oh, od) in orient[idx]:
                if ow > W or oh > H or od > D:
                    continue
                
                for (ax, ay, az) in anchors:
                    if ax + ow > W or ay + oh > H or az + od > D:
                        continue
                    
                    # Fast bitmap overlap check
                    if not can_place_bitmap(ax, ay, az, ow, oh, od):
                        continue
                    
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
                place_box_bitmap(bx, by, bz, bw, bh, bd)
                
                # Generate new anchors from the placed box
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
                        
        return filled, placed

    def make_orders():
        orders = []
        # 1. Descending volume
        orders.append(sorted(range(n), key=lambda i: -box_vol[i]))
        # 2. Ascending volume
        orders.append(sorted(range(n), key=lambda i: box_vol[i]))
        # 3. Largest min dimension
        orders.append(sorted(range(n), key=lambda i: -min(boxes[i])))
        # 4. Largest max dimension
        orders.append(sorted(range(n), key=lambda i: -max(boxes[i])))
        # 5. Descending surface area
        def surf(i):
            a, b, c = boxes[i]
            return 2 * (a * b + a * c + b * c)
        orders.append(sorted(range(n), key=lambda i: -surf(i)))
        # 6. Descending "compactness" (vol / max_dim)
        def compact(i):
            a, b, c = boxes[i]
            m = max(a, b, c)
            return (a * b * c) / m if m else 0
        orders.append(sorted(range(n), key=lambda i: -compact(i)))
        # 7. Descending aspect ratio balance (prefer cubical)
        def balance(i):
            a, b, c = sorted(boxes[i])
            if c == 0:
                return 0
            return (a * b) / c
        orders.append(sorted(range(n), key=lambda i: -balance(i)))
        # 8. Descending sum of dimensions
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
    # We perturb the order and re-pack.
    rng = random.Random(12345)
    attempts = 0
    max_attempts = 2000
    
    while time.time() < deadline and attempts < max_attempts:
        if best_order is None:
            order = list(rng.choice(base_orders))
        else:
            order = list(best_order)
            
        # Choose perturbation strategy
        strategy = rng.random()
        if strategy < 0.3:
            # Random swaps
            for _ in range(rng.randint(1, 5)):
                i, j = rng.randrange(n), rng.randrange(n)
                order[i], order[j] = order[j], order[i]
        elif strategy < 0.6:
            # Segment reversal
            i, j = rng.randrange(n), rng.randrange(n)
            if i > j:
                i, j = j, i
            order[i:j+1] = reversed(order[i:j+1])
        else:
            # Insertion
            i, j = rng.randrange(n), rng.randrange(n)
            elem = order.pop(i)
            order.insert(j, elem)
            
        filled, placed = try_pack(order, deadline)
        attempts += 1
        
        if filled > best_filled:
            best_filled = filled
            best_placed = placed
            best_order = list(order)
            
    # Reset occupancy bitmap before returning (clean state)
    occ[:] = 0
    
    # Build final result
    placement_results = [None] * n
    for (x, y, z, w, h, d, idx) in best_placed:
        placement_results[idx] = (0, x, y, z, w, h, d)
    
    for i in range(n):
        if placement_results[i] is None:
            w, h, d = boxes[i]
            placement_results[i] = (-1, 0, 0, 0, w, h, d)
    
    return placement_results
