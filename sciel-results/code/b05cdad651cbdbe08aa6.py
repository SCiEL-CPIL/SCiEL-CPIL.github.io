import numpy as np
import random
import time

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Pre-calculate volumes and sort indices by volume descending
    box_vols = [w * h * d for w, h, d in boxes]
    sorted_indices = sorted(range(n), key=lambda i: box_vols[i], reverse=True)
    
    # Pre-compute unique rotations for each box that fit in container bounds
    box_rotations = []
    for i in range(n):
        w, h, d = boxes[i]
        rots = []
        seen = set()
        for rw, rh, rd in [(w,h,d), (w,d,h), (h,w,d), (h,d,w), (d,w,h), (d,h,w)]:
            if (rw, rh, rd) in seen:
                continue
            seen.add((rw, rh, rd))
            if rw <= W and rh <= H and rd <= D:
                rots.append((rw, rh, rd))
        box_rotations.append(rots)
        
    best_vol = -1
    best_placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n)]
    
    start_time = time.time()
    max_time = 110.0
    
    # Increase restarts, but keep each restart fast by limiting candidate trials
    num_restarts = 500 
    max_candidates_to_try = 30  # Limit trials per box to keep restarts fast
    
    for r in range(num_restarts):
        if time.time() - start_time > max_time:
            break
            
        try:
            grid = np.zeros((D, H, W), dtype=bool)
        except MemoryError:
            break
            
        candidates = set([(0, 0, 0)])
        placements = [None] * n
        current_vol = 0
        
        for idx in sorted_indices:
            if time.time() - start_time > max_time:
                # Time up, return current best so far or break
                break
                
            rots = box_rotations[idx]
            if not rots:
                placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
                continue
                
            placed = False
            cand_list = list(candidates)
            
            # Strategy: Sort by proximity to origin (x+y+z) to fill corners
            # Then try top K. If fail, try random K.
            cand_list.sort(key=lambda c: c[0] + c[1] + c[2])
            
            # Try deterministic top candidates
            for cx, cy, cz in cand_list[:max_candidates_to_try]:
                if placed:
                    break
                for rw, rh, rd in rots:
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    if not grid[cz:cz+rd, cy:cy+rh, cx:cx+rw].any():
                        grid[cz:cz+rd, cy:cy+rh, cx:cx+rw] = True
                        placements[idx] = (0, cx, cy, cz, rw, rh, rd)
                        current_vol += rw * rh * rd
                        placed = True
                        
                        new_cands = [
                            (cx + rw, cy, cz),
                            (cx, cy + rh, cz),
                            (cx, cy, cz + rd),
                            (cx + rw, cy + rh, cz),
                            (cx + rw, cy, cz + rd),
                            (cx, cy + rh, cz + rd),
                            (cx + rw, cy + rh, cz + rd)
                        ]
                        for nc in new_cands:
                            if nc not in candidates:
                                candidates.add(nc)
                        break
                        
            # If not placed, try random candidates
            if not placed and len(cand_list) > max_candidates_to_try:
                random.shuffle(cand_list)
                for cx, cy, cz in cand_list[max_candidates_to_try:max_candidates_to_try*2]:
                    if placed:
                        break
                    for rw, rh, rd in rots:
                        if cx + rw > W or cy + rh > H or cz + rd > D:
                            continue
                        if not grid[cz:cz+rd, cy:cy+rh, cx:cx+rw].any():
                            grid[cz:cz+rd, cy:cy+rh, cx:cx+rw] = True
                            placements[idx] = (0, cx, cy, cz, rw, rh, rd)
                            current_vol += rw * rh * rd
                            placed = True
                            
                            new_cands = [
                                (cx + rw, cy, cz),
                                (cx, cy + rh, cz),
                                (cx, cy, cz + rd),
                                (cx + rw, cy + rh, cz),
                                (cx + rw, cy, cz + rd),
                                (cx, cy + rh, cz + rd),
                                (cx + rw, cy + rh, cz + rd)
                            ]
                            for nc in new_cands:
                                if nc not in candidates:
                                    candidates.add(nc)
                            break
                            
            if not placed:
                placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
                
        if current_vol > best_vol:
            best_vol = current_vol
            # Ensure all placements are set (in case of early break)
            for i in range(n):
                if placements[i] is None:
                    placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
            best_placements = placements

    # Final check if no placement was ever made (should not happen if n>0)
    if best_vol < 0:
        return [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        
    return best_placements
