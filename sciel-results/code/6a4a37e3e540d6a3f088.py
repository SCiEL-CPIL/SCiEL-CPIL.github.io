"""
Strategy: Skyline-based 2D Height Map Packing with Randomized Restart.
Unlike the permutation-based local search, this method discretizes the container
floor into a grid. It maintains a 2D array of minimum heights (skyline) for each
cell. Placing a box involves checking if the box fits within the container bounds
and if the current skyline heights in the covered cells are low enough. Upon
placement, the skyline heights are updated. This reduces overlap checking from
O(N_placed) to O(1) per cell, allowing for thousands of fast greedy attempts
with different random seedings and sorting strategies within the time limit.
"""
import time
import random
import math
import itertools

# EVOLVE-BLOCK-START

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    # Precompute orientations
    # We want to keep dimensions as integers if possible, but problem might have floats.
    # Assuming standard integer or simple float dims.
    all_orients = []
    for i in range(n):
        w, h, d = boxes[i]
        orients = list(set(itertools.permutations((w, h, d))))
        all_orients.append(orients)
        
    volumes = [w*h*d for w, h, d in boxes]
    container_vol = W * H * D
    
    # Grid resolution for the skyline
    # We need to balance accuracy vs speed.
    # If dimensions are integers, cell size 1 is optimal.
    # If floats, we might need to scale or use a coarser grid.
    # Let's assume inputs are "nice" (integers or multiples of 0.1).
    # For safety, let's use a grid size that is a divisor of the container dims if possible,
    # or just a fixed resolution.
    
    # Let's determine a cell size.
    # If W, H, D are integers, use 1.
    # If not, use a small epsilon or scale.
    # Given the "60-box instances" and typical coding challenge constraints, 
    # integer dimensions are likely. If not, we can scale to integers.
    
    def is_integer(x):
        return abs(x - round(x)) < 1e-9

    if all(is_integer(v) for v in container) and all(is_integer(v) for b in boxes for v in b):
        # Scale to integers for exact grid operations
        scale = 1
        W_int = int(round(W))
        H_int = int(round(H))
        D_int = int(round(D))
        boxes_int = [[int(round(w)), int(round(h)), int(round(d))] for w, h, d in boxes]
        volumes_int = [w*h*d for w, h, d in boxes_int]
        container_vol_int = W_int * H_int * D_int
        
        # Grid dimensions
        grid_w = W_int
        grid_h = H_int
        
        # Initialize skyline: grid_h x grid_w array of 0s
        # skyline[y][x] = min height at (x, y)
        
        def pack_with_skyline(order, seed):
            random.seed(seed)
            # Create a copy of the skyline? No, we need to reset it for each attempt.
            # skyline[y][x]
            skyline = [[0] * grid_w for _ in range(grid_h)]
            
            placed = []
            result = [None] * n
            filled = 0
            
            for idx in order:
                bw, bh, bd = boxes_int[idx]
                
                best_pos = None
                best_orient = None
                best_cost = float('inf')
                
                # Try all orientations
                for ow, oh, od in all_orients[idx]:
                    # Check if box fits in container
                    if ow > W_int or oh > H_int or od > D_int:
                        continue
                    
                    # We need to find a position (px, py) such that
                    # 1. px + ow <= W_int
                    # 2. py + oh <= H_int
                    # 3. For all x in [px, px+ow), y in [py, py+oh): skyline[y][x] + od <= D_int
                    #    AND we want to minimize cost (e.g., py*W_int + px + z_height)
                    
                    # Optimization: Instead of checking all px, py, we can iterate
                    # but that is O(W*H) per box per orientation.
                    # For 60 boxes, W,H ~ 20-50, this is 2500 * 60 * 6 = 900k ops per pack.
                    # 1000 restarts = 900M ops. Might be too slow in Python.
                    
                    # Better approach: "First Fit" or "Best Fit" on a smaller set of candidates?
                    # Or use a "skyline" algorithm that maintains a list of "slots".
                    
                    # Let's stick to a simpler heuristic:
                    # Iterate py from 0 to H_int - oh
                    # Iterate px from 0 to W_int - ow
                    # Check height.
                    
                    # To speed up, we can break early if we find a "good" spot.
                    # Cost: prefer lower Z (skyline height), then lower Y, then lower X.
                    # Actually, standard skyline packing prefers "lowest z, then most compact".
                    
                    found = False
                    for py in range(H_int - oh + 1):
                        if found: break
                        for px in range(W_int - ow + 1):
                            # Check if box fits in this orientation at (px, py)
                            min_z = 0
                            fits = True
                            # We need the MAX skyline height in the area to know the base Z
                            # Actually, for a box to sit on top, the base Z is max(skyline[y][x] for x,y in box area)
                            # Wait, the box must be supported? No, just no overlap.
                            # But usually "skyline" implies gravity. If we allow floating, it's harder.
                            # The task says "places ... without overlap". It does NOT explicitly say "gravity" or "supported".
                            # However, standard container loading usually implies gravity (boxes sit on floor or other boxes).
                            # If floating is allowed, we can just put everything at Z=0 if no overlap in X,Y.
                            # But then Z dimension is only about height.
                            # Given "3D Container Loading", gravity is a safe assumption for "loading".
                            # If I assume gravity, the box sits at Z = max(skyline[y][x] in area).
                            
                            max_z = 0
                            for y in range(py, py + oh):
                                row = skyline[y]
                                for x in range(px, px + ow):
                                    if row[x] > max_z:
                                        max_z = row[x]
                                        
                            if max_z + od > D_int:
                                continue
                                
                            # It fits!
                            # Cost: Prefer lower max_z (to keep things low), then lower py, then lower px
                            cost = max_z * 10000 + py * 100 + px
                            
                            if cost < best_cost:
                                best_cost = cost
                                best_pos = (px, py, max_z)
                                best_orient = (ow, oh, od)
                                found = True # We found the best for this orientation? 
                                # No, we should check all orientations.
                                # But if we found a spot with Z=0, it's hard to beat.
                                # Let's not break orientation loop immediately unless cost is 0.
                                # Actually, for speed, let's just take the first valid one found in a smart order?
                                # Or keep the loop but break inner loops if we find a "perfect" fit (Z=0, Y=0, X=0).
                                if cost == 0:
                                    break
                        # if found and cost == 0: break # Outer loop break logic is tricky here
                        # Let's just let it run to find the absolute best (px,py) for this orientation
                        # But that's O(W*H).
                        # Optimization: If we find a spot at py=0, px=0, we can break py loop?
                        # Not necessarily, another orientation might be better? No, same volume.
                        # But a different (px,py) might allow future boxes to fit better.
                        # "Best Fit" (lowest Z, then compact) is a good heuristic.
                        
                    # End orientation loop
                    if best_pos is not None and best_cost == 0:
                        break # Can't get better than 0,0,0
                
                if best_pos is not None:
                    px, py, pz = best_pos
                    ow, oh, od = best_orient
                    placed.append((px, py, pz, ow, oh, od))
                    result[idx] = (0, px, py, pz, ow, oh, od)
                    filled += ow * oh * od
                    
                    # Update skyline
                    for y in range(py, py + oh):
                        row = skyline[y]
                        for x in range(px, px + ow):
                            if row[x] < pz + od:
                                row[x] = pz + od
                else:
                    result[idx] = (-1, 0, 0, 0, bw, bh, bd)
                    
            return filled, result

    else:
        # Fallback for non-integer dims: Scale to integers by 10 or 100
        # Detect decimal places
        def get_scale_factor(dims):
            max_dec = 0
            for v in dims:
                s = str(v)
                if '.' in s:
                    dec = len(s.split('.')[1])
                    if dec > max_dec: max_dec = dec
            return 10 ** max_dec

        scale = get_scale_factor(container + [v for b in boxes for v in b])
        W_int = int(round(W * scale))
        H_int = int(round(H * scale))
        D_int = int(round(D * scale))
        boxes_int = [[int(round(w * scale)), int(round(h * scale)), int(round(d * scale))] for w, h, d in boxes]
        volumes_int = [w*h*d for w, h, d in boxes_int]
        container_vol_int = W_int * H_int * D_int
        
        grid_w = W_int
        grid_h = H_int
        
        def pack_with_skyline(order, seed):
            random.seed(seed)
            skyline = [[0] * grid_w for _ in range(grid_h)]
            
            placed = []
            result = [None] * n
            filled = 0
            
            for idx in order:
                bw, bh, bd = boxes_int[idx]
                
                best_pos = None
                best_orient = None
                best_cost = float('inf')
                
                for ow, oh, od in all_orients[idx]:
                    if ow > W_int or oh > H_int or od > D_int:
                        continue
                    
                    found = False
                    for py in range(H_int - oh + 1):
                        if found and best_cost == 0: break
                        for px in range(W_int - ow + 1):
                            max_z = 0
                            fits = True
                            for y in range(py, py + oh):
                                row = skyline[y]
                                for x in range(px, px + ow):
                                    if row[x] > max_z:
                                        max_z = row[x]
                                        
                            if max_z + od > D_int:
                                continue
                                
                            cost = max_z * 10000 + py * 100 + px
                            
                            if cost < best_cost:
                                best_cost = cost
                                best_pos = (px, py, max_z)
                                best_orient = (ow, oh, od)
                                if cost == 0:
                                    found = True
                                    break
                        if found and best_cost == 0:
                            break
                    
                    if best_pos is not None and best_cost == 0:
                        break
                
                if best_pos is not None:
                    px, py, pz = best_pos
                    ow, oh, od = best_orient
                    placed.append((px, py, pz, ow, oh, od))
                    result[idx] = (0, px, py, pz, ow, oh, od)
                    filled += ow * oh * od
                    
                    for y in range(py, py + oh):
                        row = skyline[y]
                        for x in range(px, px + ow):
                            if row[x] < pz + od:
                                row[x] = pz + od
                else:
                    result[idx] = (-1, 0, 0, 0, bw, bh, bd)
                    
            return filled, result

    # Strategies for ordering
    # 1. Volume Descending
    # 2. Volume Ascending
    # 3. Max Dimension Descending
    # 4. Random
    
    best_filled = 0
    best_result = None
    
    # Generate a few static orders
    orders = [
        sorted(range(n), key=lambda i: volumes_int[i], reverse=True),
        sorted(range(n), key=lambda i: volumes_int[i]),
        sorted(range(n), key=lambda i: max(boxes_int[i]), reverse=True),
        sorted(range(n), key=lambda i: min(boxes_int[i]), reverse=True)
    ]
    
    # Evaluate static orders
    for i, order in enumerate(orders):
        if time.time() > deadline:
            break
        filled, result = pack_with_skyline(order, i)
        if filled > best_filled:
            best_filled = filled
            best_result = result
            
    # Randomized Restart with Random Orders
    # We can also shuffle the static orders slightly
    attempts = 0
    max_attempts = 2000 # Limit by time, not count
    
    while time.time() < deadline:
        # Pick a base strategy
        base_order = orders[random.randint(0, len(orders)-1)][:]
        
        # Perturb: Swap 2-5 elements
        for _ in range(random.randint(2, 5)):
            i, j = random.sample(range(n), 2)
            base_order[i], base_order[j] = base_order[j], base_order[i]
            
        filled, result = pack_with_skyline(base_order, attempts)
        attempts += 1
        
        if filled > best_filled:
            best_filled = filled
            best_result = result
            
    # Final check
    if best_result is None:
        # Fallback: Place nothing or first box?
        # If no box fits, return all -1
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    # Scale back results to original units if we scaled
    if scale != 1:
        scaled_result = []
        for r in best_result:
            bin_id, x, y, z, w, h, d = r
            if bin_id == -1:
                scaled_result.append(r)
            else:
                sx = x / scale
                sy = y / scale
                sz = z / scale
                sw = w / scale
                sh = h / scale
                sd = d / scale
                # Ensure precision
                scaled_result.append((bin_id, sx, sy, sz, sw, sh, sd))
        return scaled_result
        
    return best_result

# EVOLVE-BLOCK-END
