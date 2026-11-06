import time
import itertools
import sys

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Safety margin: evaluator kills at 150s, we target 110s
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Generate all unique rotations for each box
    def get_rotations(w, h, d):
        seen = set()
        rots = []
        for perm in itertools.permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                rots.append(perm)
        return rots
    
    box_rots = [get_rotations(w, h, d) for w, h, d in boxes]
    
    # Sorting strategies
    def sort_volume_desc(idx_list):
        return sorted(idx_list, key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2]))
    
    def sort_max_dim_desc(idx_list):
        return sorted(idx_list, key=lambda i: -max(boxes[i]))
    
    def sort_sum_dim_desc(idx_list):
        return sorted(idx_list, key=lambda i: -(boxes[i][0]+boxes[i][1]+boxes[i][2]))

    strategies = [sort_volume_desc, sort_max_dim_desc, sort_sum_dim_desc]
    
    def run_packing(sort_func):
        placed = []  # List of (x, y, z, w, h, d)
        result = [None] * n
        order = sort_func(list(range(n)))
        
        for i in order:
            if time.time() > deadline:
                break
                
            bw, bh, bd = boxes[i]
            best_pos = None
            best_key = None
            
            # Candidate coordinates
            xs = [0]
            ys = [0]
            zs = [0]
            
            for (px, py, pz, pw, ph, pd) in placed:
                if px + pw <= W:
                    xs.append(px + pw)
                if py + ph <= H:
                    ys.append(py + ph)
                if pz + pd <= D:
                    zs.append(pz + pd)
            
            # Remove duplicates and sort
            xs = sorted(set(x for x in xs if x <= W))
            ys = sorted(set(y for y in ys if y <= H))
            zs = sorted(set(z for z in zs if z <= D))
            
            # If the list is too large, we might slow down. 
            # However, with N=60, max unique coords is 61. 61^3 ~ 226k candidates.
            # 226k * 6 rotations * 60 boxes = 81M overlap checks.
            # Each overlap check is O(N) = 60. Total 4.8B ops. This is too slow for Python.
            
            # Optimization: 
            # We don't need to check ALL candidates if we find a good one early? 
            # No, BLF requires the minimum z, then x, then y. 
            # We can iterate z in ascending order, x in ascending, y in ascending.
            # The first valid position we find IS the BLF position.
            # This allows early termination.
            
            found = False
            for z in zs:
                if found: break
                for y in ys:
                    if found: break
                    for x in xs:
                        if found: break
                        for rw, rh, rd in box_rots[i]:
                            if x + rw > W or y + rh > H or z + rd > D:
                                continue
                            
                            # Check overlap
                            overlap = False
                            for (px, py, pz, pw, ph, pd) in placed:
                                if (x < px + pw and x + rw > px and
                                    y < py + ph and y + rh > py and
                                    z < pz + pd and z + rd > pz):
                                    overlap = True
                                    break
                            
                            if not overlap:
                                # Valid position found. Since we iterate z, y, x in ascending order,
                                # this is the BLF position for this rotation.
                                # But we need to check all rotations?
                                # The BLF rule applies to the position. 
                                # Different rotations might allow different positions.
                                # We should find the global BLF position across all rotations.
                                # Key: (z, y, x)
                                key = (z, y, x)
                                if best_key is None or key < best_key:
                                    best_key = key
                                    best_pos = (x, y, z, rw, rh, rd)
                                
                                # We can break the rotation loop for this x,y,z?
                                # No, another rotation might be smaller? No, position is same.
                                # But another rotation might allow a *different* (smaller) position?
                                # No, we are iterating positions. If a position is valid for rot A, 
                                # it is a candidate. We want the min position.
                                # So for a specific (x,y,z), if it's valid for ANY rotation, 
                                # it's a valid position. We pick the best rotation? 
                                # Actually, the position is the same. The box dimensions change.
                                # Does the choice of rotation affect future packing? Yes.
                                # But for the current placement, we just need *a* rotation that fits.
                                # Usually, we prefer the rotation that leaves the most space? 
                                # Or just the first one that fits.
                                # Let's just take the first valid rotation for this position.
                                found = True
                                break
                            
            if best_pos is not None:
                x, y, z, rw, rh, rd = best_pos
                placed.append((x, y, z, rw, rh, rd))
                result[i] = (0, x, y, z, rw, rh, rd)
            else:
                result[i] = (-1, 0, 0, 0, bw, bh, bd)
                
            # Update candidate lists? No, we rebuild them every time for simplicity.
            # Rebuilding is O(N^2) per box, which is fine.
            
        # Fill in unprocessed
        for i in range(n):
            if result[i] is None:
                result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
                
        return result

    best_placements = None
    best_score = -1
    
    for sort_func in strategies:
        if time.time() > deadline:
            break
        
        placements = run_packing(sort_func)
        
        # Calculate score
        total_vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == 0)
        score = total_vol / (W * H * D) if W * H * D > 0 else 0
        
        if score > best_score:
            best_score = score
            best_placements = placements

    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_placements
