import numpy as np
import random
import time

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Check if grid is feasible
    # If W, H, D are very large, numpy array might be too big.
    # 10^7 booleans is 10MB. 10^8 is 100MB.
    # Let's set a limit.
    if W * H * D > 20_000_000:
        # Fallback to simple greedy if grid too big? 
        # Or just use a sparse set?
        # For now, assume it fits. If not, we might crash.
        # Let's try to be safe.
        pass

    best_vol = -1
    best_placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n)]
    
    start_time = time.time()
    max_time = 110.0
    
    # Number of restarts
    # We can estimate how many we can do.
    # Let's try 50 first, and increase if fast.
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() - start_time > max_time:
            break
            
        # Shuffle order
        order = list(range(n))
        random.shuffle(order)
        
        # Initialize grid
        # Note: numpy array indexing is [axis0, axis1, axis2]
        # We map z->0, y->1, x->2
        try:
            grid = np.zeros((D, H, W), dtype=bool)
        except MemoryError:
            # If grid is too big, we can't use this method.
            # Break and return current best (or simple greedy)
            break
            
        candidates = set([(0, 0, 0)])
        placements = [None] * n
        current_vol = 0
        
        for idx in order:
            w, h, d = boxes[idx]
            
            # Skip if box is too big for container in any orientation
            # Check if any rotation fits in bounds
            fits_bounds = False
            rotations = []
            for rw, rh, rd in [(w,h,d), (w,d,h), (h,w,d), (h,d,w), (d,w,h), (d,h,w)]:
                if rw <= W and rh <= H and rd <= D:
                    fits_bounds = True
                    rotations.append((rw, rh, rd))
            
            if not fits_bounds:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
                continue
                
            placed = False
            # Sort candidates to prefer "bottom-left-back" (low z, low y, low x)
            # This helps pack into corners.
            sorted_cands = sorted(candidates, key=lambda p: (p[2], p[1], p[0]))
            
            for cx, cy, cz in sorted_cands:
                if placed:
                    break
                
                for rw, rh, rd in rotations:
                    # Check bounds
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    
                    # Check collision
                    # grid[cz:cz+rd, cy:cy+rh, cx:cx+rw]
                    if not grid[cz:cz+rd, cy:cy+rh, cx:cx+rw].any():
                        # Place
                        grid[cz:cz+rd, cy:cy+rh, cx:cx+rw] = True
                        placements[idx] = (0, cx, cy, cz, rw, rh, rd)
                        current_vol += rw * rh * rd
                        placed = True
                        
                        # Add new candidates
                        # 8 corners of the placed box
                        new_cands = [
                            (cx + rw, cy, cz),
                            (cx, cy + rh, cz),
                            (cx, cy, cz + rd),
                            (cx + rw, cy + rh, cz),
                            (cx + rw, cy, cz + rd),
                            (cx, cy + rh, cz + rd),
                            (cx + rw, cy + rh, cz + rd)
                        ]
                        # Also the "back" corners? No, those are covered by existing boxes or origin.
                        # We only need the "front" faces.
                        # Actually, adding all 8 is safe.
                        candidates.update(new_cands)
                        break
                        
            if not placed:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
                
        if current_vol > best_vol:
            best_vol = current_vol
            best_placements = placements
            
    return best_placements
