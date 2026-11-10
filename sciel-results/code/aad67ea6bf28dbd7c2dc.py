import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    num_points = q * q
    num_directions = q + 1
    k = 3
    
    def add_f49(a, b, c, d):
        return ((a + c) % p, (b + d) % p)
        
    def mul_f49(a, b, c, d):
        return ((a * c + b * d * k) % p, (a * d + b * c) % p)
        
    def get_idx(xa, xb, ya, yb):
        return (xa * p + xb) * q + (ya * p + yb)
        
    def get_coords(idx):
        y_part = idx % q
        x_part = idx // q
        yb = y_part % p
        ya = y_part // p
        xb = x_part % p
        xa = x_part // p
        return xa, xb, ya, yb

    lines = []
    point_lines = [[] for _ in range(num_points)]
    line_idx = 0
    
    for xa in range(p):
        for xb in range(p):
            line_pts = []
            for ya in range(p):
                for yb in range(p):
                    idx = get_idx(xa, xb, ya, yb)
                    line_pts.append(idx)
            lines.append(line_pts)
            for idx in line_pts:
                point_lines[idx].append(line_idx)
            line_idx += 1
            
    for ma in range(p):
        for mb in range(p):
            for ya_int in range(p):
                for yb_int in range(p):
                    line_pts = []
                    for t_a in range(p):
                        for t_b in range(p):
                            xa = t_a
                            xb = t_b
                            ya, yb = add_f49(ya_int, yb_int, *mul_f49(t_a, t_b, ma, mb))
                            idx = get_idx(xa, xb, ya, yb)
                            line_pts.append(idx)
                    lines.append(line_pts)
                    for idx in line_pts:
                        point_lines[idx].append(line_idx)
                    line_idx += 1
                        
    def try_greedy():
        in_C = [False] * num_points
        clean_mask = [(1 << 50) - 1] * num_points
        C_size = 0
        candidates = list(range(num_points))
        random.shuffle(candidates)
        
        while candidates:
            x = candidates.pop(0)
            if in_C[x]:
                continue
            if clean_mask[x] == 0:
                continue
            
            valid_add = True
            updates = []
            
            for li in point_lines[x]:
                bit = 1 << li
                for y in lines[li]:
                    if y == x:
                        continue
                    if clean_mask[y] & bit:
                        new_mask = clean_mask[y] & ~bit
                        if new_mask == 0:
                            valid_add = False
                            break
                        updates.append((y, new_mask))
                if not valid_add:
                    break
            
            if valid_add:
                for y, new_mask in updates:
                    clean_mask[y] = new_mask
                in_C[x] = True
                C_size += 1
        
        points = []
        for i in range(num_points):
            if not in_C[i]:
                xa, xb, ya, yb = get_coords(i)
                points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64), C_size

    best_C_size = 0
    best_result = None
    start_time = time.time()
    deadline = start_time + 590
    
    while time.time() < deadline:
        res, size = try_greedy()
        if size > best_C_size:
            best_C_size = size
            best_result = res
            
    if best_result is None:
        # Fallback: return empty set? No, must be valid.
        # Return full set (N = all points, C = empty)
        points = []
        for i in range(num_points):
            xa, xb, ya, yb = get_coords(i)
            points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)
        
    return best_result
