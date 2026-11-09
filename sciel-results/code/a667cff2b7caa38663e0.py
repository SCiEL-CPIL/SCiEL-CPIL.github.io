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
    
    # Pre-compute unique rotations for each box that fit in the container
    box_rotations = []
    for w, h, d in boxes:
        seen = set()
        rots = []
        for rw, rh, rd in [(w,h,d), (w,d,h), (h,w,d), (h,d,w), (d,w,h), (d,h,w)]:
            if (rw, rh, rd) not in seen:
                seen.add((rw, rh, rd))
                if rw <= W and rh <= H and rd <= D:
                    rots.append((rw, rh, rd))
        box_rotations.append(rots)
        
    best_vol = -1
    best_placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n)]
    
    start_time = time.time()
    max_time = 105.0
    
    # Run as many restarts as possible within the time budget
    while True:
        if time.time() - start_time > max_time:
            break
            
        try:
            grid = np.zeros((D, H, W), dtype=bool)
        except MemoryError:
            break
            
        candidates = [(0, 0, 0)]
        cand_set = {(0, 0, 0)}
        placements = [None] * n
        current_vol = 0
        
        # Process boxes in sorted order (largest first)
        for idx in sorted_indices:
            w, h, d = boxes[idx]
            rots = box_rotations[idx]
            
            if not rots:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
                continue
                
            placed = False
            
            # Shuffle candidates to introduce diversity
            # Limit candidate checks if the list grows too large to maintain speed
            # This is a heuristic to keep each restart fast
            if len(candidates) > 200:
                # Randomly sample a subset of candidates to check
                # This allows exploring different parts of the space without checking all
                random.shuffle(candidates)
                check_list = candidates[:200]
            else:
                random.shuffle(candidates)
                check_list = candidates
            
            for cx, cy, cz in check_list:
                if placed:
                    break
                
                for rw, rh, rd in rots:
                    # Check bounds
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    
                    # Check collision
                    if not grid[cz:cz+rd, cy:cy+rh, cx:cx+rw].any():
                        # Place
                        grid[cz:cz+rd, cy:cy+rh, cx:cx+rw] = True
                        placements[idx] = (0, cx, cy, cz, rw, rh, rd)
                        current_vol += rw * rh * rd
                        placed = True
                        
                        # Add new candidates
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
                            nx, ny, nz = nc
                            if 0 <= nx < W and 0 <= ny < H and 0 <= nz < D:
                                if nc not in cand_set:
                                    cand_set.add(nc)
                                    candidates.append(nc)
                        break
                        
            if not placed:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
                
        if current_vol > best_vol:
            best_vol = current_vol
            best_placements = placements

    return best_placements
