import time
import itertools

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 115.0 # Safety margin
    
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    
    # Helper to check overlap
    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        if x1 + w1 <= x2 or x2 + w2 <= x1: return False
        if y1 + h1 <= y2 or y2 + h2 <= y1: return False
        if z1 + d1 <= z2 or z2 + d2 <= z1: return False
        return True

    def get_orientations(w, h, d):
        # Unique orientations to avoid redundant checks
        # (w,h,d), (w,d,h), (h,w,d), (h,d,w), (d,w,h), (d,h,w)
        # But we can just generate all permutations and filter unique ones
        perms = set(itertools.permutations([w, h, d]))
        return list(perms)

    def greedy_pack(sort_key):
        # Sort boxes based on key
        indexed_boxes = sorted(enumerate(boxes), key=lambda x: sort_key(x[1]), reverse=True)
        
        placed = [] # List of (idx, x, y, z, w, h, d)
        corners = [(0, 0, 0)]
        best_vol = 0
        
        for idx, (w, h, d) in indexed_boxes:
            orientations = get_orientations(w, h, d)
            best_placement = None
            best_score = float('inf') # We want to minimize x+y+z
            
            # Try each orientation
            for ow, oh, od in orientations:
                # Try each corner
                for cx, cy, cz in corners:
                    # Check bounds
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    
                    # Check overlap with placed boxes
                    valid = True
                    for _, px, py, pz, pw, ph, pd in placed:
                        if overlaps(cx, cy, cz, ow, oh, od, px, py, pz, pw, ph, pd):
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Score: distance to origin (lower is better)
                    score = cx + cy + cz
                    # Tie-breaker: maybe prefer placing larger boxes deeper? 
                    # Or just keep first found if score is same.
                    if score < best_score:
                        best_score = score
                        best_placement = (idx, cx, cy, cz, ow, oh, od)
            
            if best_placement:
                placed.append(best_placement)
                best_vol += best_placement[4] * best_placement[5] * best_placement[6]
                
                # Generate new corners
                _, bx, by, bz, bw, bh, bd = best_placement
                new_corners = [
                    (bx + bw, by, bz),
                    (bx, by + bh, bz),
                    (bx, by, bz + bd)
                ]
                for nc in new_corners:
                    if nc not in corners:
                        corners.append(nc)
            else:
                # Box cannot be placed
                pass
        
        # Construct result
        result = [(-1, 0, 0, 0, 0, 0, 0)] * n
        for idx, x, y, z, w, h, d in placed:
            result[idx] = (0, x, y, z, w, h, d)
        
        return result, best_vol

    # Strategy 1: Sort by Volume
    res1, vol1 = greedy_pack(lambda b: b[0]*b[1]*b[2])
    if time.time() - start_time > DEADLINE:
        return res1

    # Strategy 2: Sort by Max Dimension
    res2, vol2 = greedy_pack(lambda b: max(b))
    if time.time() - start_time > DEADLINE:
        return res2 if vol2 > vol1 else res1

    # Strategy 3: Sort by Sum of Dimensions
    res3, vol3 = greedy_pack(lambda b: sum(b))
    
    best_res = res1
    best_vol = vol1
    if vol2 > best_vol:
        best_res = res2
        best_vol = vol2
    if vol3 > best_vol:
        best_res = res3
        best_vol = vol3
        
    return best_res
