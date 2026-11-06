"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

def load(boxes, container):
    W, H, D = container
    W, H, D = int(W), int(H), int(D)
    n = len(boxes)
    
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    deadline = time.time() + 115.0  # Safety margin for 120s limit
    
    # Pre-compute unique orientations for each box
    def get_orientations(box):
        w, h, d = box
        orients = set()
        for perm in itertools.permutations([w, h, d]):
            orients.add(perm)
        return list(orients)
    
    orientations = [get_orientations(b) for b in boxes]
    
    # 3D Occupancy Grid
    use_grid = (W * H * D) < 50_000_000
    if use_grid:
        grid = np.zeros((W, H, D), dtype=np.int8)
    else:
        grid = None
        
    def is_valid_grid(x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        if grid is not None:
            if np.any(grid[x:x+w, y:y+h, z:z+d] > 0):
                return False
            return True
        else:
            return False

    def place_in_grid(x, y, z, w, h, d):
        if grid is not None:
            grid[x:x+w, y:y+h, z:z+d] = 1

    def unplace_in_grid(x, y, z, w, h, d):
        if grid is not None:
            grid[x:x+w, y:y+h, z:z+d] = 0

    def pack_with_order(order):
        placed = []
        result = [None] * n
        current_time = time.time()
        
        for idx in order:
            if current_time > deadline:
                break
            box = boxes[idx]
            orients = orientations[idx]
            
            # Candidate points: corners of already placed boxes + origin
            candidates = [(0, 0, 0)]
            for pb in placed:
                x, y, z, w, h, d = pb
                candidates.append((x, y, z))
                candidates.append((x + w, y, z))
                candidates.append((x, y + h, z))
                candidates.append((x, y, z + d))
                candidates.append((x + w, y + h, z))
                candidates.append((x + w, y, z + d))
                candidates.append((x, y + h, z + d))
                candidates.append((x + w, y + h, z + d))
            
            unique_candidates = set(candidates)
            
            best_pos = None
            best_score = float('inf')
            
            for cx, cy, cz in unique_candidates:
                for ow, oh, od in orients:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    
                    if is_valid_grid(cx, cy, cz, ow, oh, od):
                        # Score: prefer positions closer to origin
                        score = cx + cy + cz
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz, ow, oh, od)
            
            if best_pos:
                placed.append(best_pos)
                result[idx] = best_pos
                place_in_grid(*best_pos)
                current_time = time.time()
            else:
                result[idx] = None
        
        return result
    
    # Define sorting orders
    order_vol = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    order_max = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    order_aspect = sorted(range(n), key=lambda i: max(boxes[i]) / min(boxes[i]))
    order_surf = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    
    orders = [order_vol, order_max, order_aspect, order_surf]
    
    best_result = None
    best_volume = 0
    
    for order in orders:
        if time.time() > deadline:
            break
        
        if grid is not None:
            grid[:] = 0
            
        res = pack_with_order(order)
        
        vol = 0
        for i in range(n):
            if res[i] is not None:
                x, y, z, w, h, d = res[i]
                vol += w * h * d
        
        if vol > best_volume:
            best_volume = vol
            best_result = res
    
    if best_result is None:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    final_placements = []
    for i in range(n):
        if best_result[i] is not None:
            x, y, z, w, h, d = best_result[i]
            final_placements.append((0, x, y, z, w, h, d))
        else:
            w, h, d = boxes[i]
            final_placements.append((-1, 0, 0, 0, w, h, d))
            
    return final_placements

# EVOLVE-BLOCK-END
