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
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    def get_rotations(w, h, d):
        seen = set()
        rots = []
        for perm in itertools.permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                rots.append(perm)
        return rots
    
    box_rots = [get_rotations(w, h, d) for w, h, d in boxes]
    
    def sort_volume_desc(idx_list):
        return sorted(idx_list, key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2]))
    
    def sort_max_dim_desc(idx_list):
        return sorted(idx_list, key=lambda i: -max(boxes[i]))
    
    def sort_sum_dim_desc(idx_list):
        return sorted(idx_list, key=lambda i: -(boxes[i][0]+boxes[i][1]+boxes[i][2]))
    
    def sort_min_dim_asc(idx_list):
        return sorted(idx_list, key=lambda i: min(boxes[i]))
    
    def sort_ratio_desc(idx_list):
        return sorted(idx_list, key=lambda i: -(max(boxes[i])/min(boxes[i]) if min(boxes[i]) > 0 else 100))
    
    strategies = [sort_volume_desc, sort_max_dim_desc, sort_sum_dim_desc, sort_min_dim_asc, sort_ratio_desc]
    
    def run_packing(sort_func):
        placed = []
        result = [None] * n
        order = sort_func(list(range(n)))
        
        for i in order:
            if time.time() > deadline:
                break
                
            bw, bh, bd = boxes[i]
            best_pos = None
            best_key = None
            
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
            
            xs = sorted(set(x for x in xs if x <= W))
            ys = sorted(set(y for y in ys if y <= H))
            zs = sorted(set(z for z in zs if z <= D))
            
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
                            
                            overlap = False
                            for (px, py, pz, pw, ph, pd) in placed:
                                if (x < px + pw and x + rw > px and
                                    y < py + ph and y + rh > py and
                                    z < pz + pd and z + rd > pz):
                                    overlap = True
                                    break
                            
                            if not overlap:
                                key = (z, y, x)
                                if best_key is None or key < best_key:
                                    best_key = key
                                    best_pos = (x, y, z, rw, rh, rd)
                                
                                found = True
                                break
                            
            if best_pos is not None:
                x, y, z, rw, rh, rd = best_pos
                placed.append((x, y, z, rw, rh, rd))
                result[i] = (0, x, y, z, rw, rh, rd)
            else:
                result[i] = (-1, 0, 0, 0, bw, bh, bd)
                
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
        
        total_vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == 0)
        score = total_vol / (W * H * D) if W * H * D > 0 else 0
        
        if score > best_score:
            best_score = score
            best_placements = placements

    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_placements
