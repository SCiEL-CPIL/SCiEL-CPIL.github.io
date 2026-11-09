import numpy as np
import random
import time

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Pre-calculate volumes and sort indices by volume descending
    # Placing larger boxes first generally leads to better packing density
    box_vols = [w * h * d for w, h, d in boxes]
    sorted_indices = sorted(range(n), key=lambda i: box_vols[i], reverse=True)
    
    best_vol = -1
    best_placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n)]
    
    start_time = time.time()
    max_time = 110.0
    
    # Increased restarts to explore more of the solution space
    # The randomization in candidate selection allows for more diverse packings
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() - start_time > max_time:
            break
            
        # Initialize grid
        try:
            grid = np.zeros((D, H, W), dtype=bool)
        except MemoryError:
            break
            
        candidates = set([(0, 0, 0)])
        placements = [None] * n
        current_vol = 0
        
        # Process boxes in sorted order (largest first)
        for idx in sorted_indices:
            w, h, d = boxes[idx]
            
            # Check if box fits in bounds in any rotation
            fits_bounds = False
            rotations = []
            seen_rots = set()
            for rw, rh, rd in [(w,h,d), (w,d,h), (h,w,d), (h,d,w), (d,w,h), (d,h,w)]:
                if (rw, rh, rd) in seen_rots:
                    continue
                seen_rots.add((rw, rh, rd))
                if rw <= W and rh <= H and rd <= D:
                    fits_bounds = True
                    rotations.append((rw, rh, rd))
            
            if not fits_bounds:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
                continue
                
            placed = False
            
            # Shuffle candidates to introduce diversity in placement position
            # This prevents the algorithm from always filling the exact same "greedy" corners
            cand_list = list(candidates)
            random.shuffle(cand_list)
            
            for cx, cy, cz in cand_list:
                if placed:
                    break
                
                for rw, rh, rd in rotations:
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
                        
                        # Add new candidates (front corners)
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
                placements[idx] = (-1, 0, 0, 0, w, h, d)
                
        if current_vol > best_vol:
            best_vol = current_vol
            best_placements = placements
            
    return best_placements
