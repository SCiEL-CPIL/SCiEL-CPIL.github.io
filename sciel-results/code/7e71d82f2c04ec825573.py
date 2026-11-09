import numpy as np
import random
import time

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Pre-compute valid rotations for each box
    box_rotations = []
    for i in range(n):
        w, h, d = boxes[i]
        rots = []
        for rw, rh, rd in [(w,h,d), (w,d,h), (h,w,d), (h,d,w), (d,w,h), (d,h,w)]:
            if rw <= W and rh <= H and rd <= D:
                if not rots or (rw, rh, rd) not in rots:
                    rots.append((rw, rh, rd))
        box_rotations.append(rots)
    
    # Check if grid is feasible
    if W * H * D > 50_000_000:
        # Fallback: simple greedy without grid
        return _simple_greedy(boxes, container)
    
    best_vol = -1
    best_placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n)]
    
    start_time = time.time()
    max_time = 110.0
    
    # Adaptive number of restarts
    # Estimate time per restart and adjust
    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() - start_time > max_time:
            break
        
        # Different random strategies for diversity
        if r % 5 == 0:
            # Volume descending
            order = list(range(n))
            random.shuffle(order)
            order.sort(key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2])
        elif r % 5 == 1:
            # Random
            order = list(range(n))
            random.shuffle(order)
        elif r % 5 == 2:
            # Size ascending (small first)
            order = list(range(n))
            random.shuffle(order)
            order.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2])
        elif r % 5 == 3:
            # Max dimension descending
            order = list(range(n))
            random.shuffle(order)
            order.sort(key=lambda i: -max(boxes[i]))
        else:
            # Mixed: half volume desc, half random
            order1 = list(range(n // 2))
            order2 = list(range(n // 2, n))
            random.shuffle(order1)
            random.shuffle(order2)
            order1.sort(key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2])
            order = order1 + order2
            random.shuffle(order)
        
        # Initialize grid
        try:
            grid = np.zeros((D, H, W), dtype=bool)
        except MemoryError:
            break
        
        placements = [None] * n
        current_vol = 0
        
        # Use a set of candidate positions - start with origin
        candidates = [(0, 0, 0)]
        
        for idx in order:
            rots = box_rotations[idx]
            if not rots:
                placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
                continue
            
            placed = False
            # Sort candidates to prefer bottom-left-back
            candidates.sort(key=lambda p: (p[2], p[1], p[0]))
            
            best_pos = None
            best_rot = None
            best_score = -1
            
            for cx, cy, cz in candidates:
                for rw, rh, rd in rots:
                    # Check bounds
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    
                    # Check collision using grid slice
                    if not grid[cz:cz+rd, cy:cy+rh, cx:cx+rw].any():
                        # Score this position - prefer compact packing
                        # Lower z, y, x is better; also prefer positions with more "support"
                        score = -(cz * 1000 + cy * 100 + cx)
                        # Bonus for being on floor or against walls
                        if cz == 0:
                            score += 50
                        if cx == 0:
                            score += 30
                        if cy == 0:
                            score += 30
                        
                        if score > best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_rot = (rw, rh, rd)
                
                if placed:
                    break
            
            if best_pos is not None:
                cx, cy, cz = best_pos
                rw, rh, rd = best_rot
                grid[cz:cz+rd, cy:cy+rh, cx:cx+rw] = True
                placements[idx] = (0, cx, cy, cz, rw, rh, rd)
                current_vol += rw * rh * rd
                placed = True
                
                # Generate new candidate positions from the placed box's "extreme" corners
                # These are positions where a new box could potentially fit
                new_cands = set()
                # Corner at +x face
                new_cands.add((cx + rw, cy, cz))
                # Corner at +y face  
                new_cands.add((cx, cy + rh, cz))
                # Corner at +z face
                new_cands.add((cx, cy, cz + rd))
                # Combined corners
                new_cands.add((cx + rw, cy + rh, cz))
                new_cands.add((cx + rw, cy, cz + rd))
                new_cands.add((cx, cy + rh, cz + rd))
                new_cands.add((cx + rw, cy + rh, cz + rd))
                
                # Filter out positions that are completely blocked or out of bounds
                valid_new_cands = []
                for nx, ny, nz in new_cands:
                    if nx < W and ny < H and nz < D:
                        # Check if this position is actually free (at least one cell)
                        if not grid[nz, ny, nx]:
                            valid_new_cands.append((nx, ny, nz))
                
                candidates.extend(valid_new_cands)
            else:
                placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
        
        if current_vol > best_vol:
            best_vol = current_vol
            best_placements = list(placements)
    
    # Ensure all placements are set
    result = []
    for i in range(n):
        if best_placements[i] is None:
            result.append((-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
        else:
            result.append(best_placements[i])
    
    return result

def _simple_greedy(boxes, container):
    """Fallback for very large containers"""
    W, H, D = container
    n = len(boxes)
    placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    # Simple: place in order, first fit
    placed_boxes = []
    for i in range(n):
        w, h, d = boxes[i]
        rots = [(w,h,d), (w,d,h), (h,w,d), (h,d,w), (d,w,h), (d,h,w)]
        rots = [r for r in rots if r[0] <= W and r[1] <= H and r[2] <= D]
        if not rots:
            continue
        
        for rw, rh, rd in rots:
            found = False
            # Try grid positions
            for x in range(0, W - rw + 1, max(1, rw // 4)):
                for y in range(0, H - rh + 1, max(1, rh // 4)):
                    for z in range(0, D - rd + 1, max(1, rd // 4)):
                        # Check overlap
                        overlap = False
                        for px, py, pz, pw, ph, pd in placed_boxes:
                            if not (x + rw <= px or px + pw <= x or
                                    y + rh <= py or py + ph <= y or
                                    z + rd <= pz or pz + pd <= z):
                                overlap = True
                                break
                        if not overlap:
                            placements[i] = (0, x, y, z, rw, rh, rd)
                            placed_boxes.append((x, y, z, rw, rh, rd))
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
    
    return placements
