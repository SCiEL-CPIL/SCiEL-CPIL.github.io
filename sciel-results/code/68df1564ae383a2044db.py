import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k = 3

    def add_pt(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def mul_pt(a, b):
        res_a = (a[0]*b[0] + a[1]*b[1]*k) % p
        res_b = (a[0]*b[1] + a[1]*b[0]) % p
        return (res_a, res_b)

    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append(((xa, xb), (ya, yb)))
                    
    point_to_idx = {pt: i for i, pt in enumerate(points)}
    
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 1), (0, 0)))
    
    all_lines = set()
    for v in directions:
        for x in points:
            line_pts = set()
            for t_a in range(p):
                for t_b in range(p):
                    t = (t_a, t_b)
                    tv = mul_pt(t, v)
                    p_on_line = add_pt(x, tv)
                    line_pts.add(p_on_line)
            all_lines.add(frozenset(line_pts))
            
    lines_list = []
    for line in all_lines:
        idxs = [point_to_idx[pt] for pt in line]
        lines_list.append(idxs)
        
    lines_through_point = [[] for _ in range(q*q)]
    for i, line_idxs in enumerate(lines_list):
        for idx in line_idxs:
            lines_through_point[idx].append(i)
            
    # Convert to sets for faster intersection
    lines_through_set = [set(lst) for lst in lines_through_point]
    
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u*u)%p, v))
            
    C = set()
    for i, pt in enumerate(points):
        x, y = pt
        xa, xb = x
        ya, yb = y
        trace_x = (2*xa) % p
        norm_y = (ya*ya - k*yb*yb) % p
        in_unital = (trace_x + norm_y) % p == 0
        in_y_special = (ya, yb) in y_special
        if in_unital and not in_y_special:
            C.add(i)
            
    valid_lines = [set() for _ in range(q*q)]
    for x_idx in range(q*q):
        for l_idx in lines_through_point[x_idx]:
            is_valid = True
            for p_idx in lines_list[l_idx]:
                if p_idx == x_idx:
                    continue
                if p_idx in C:
                    is_valid = False
                    break
            if is_valid:
                valid_lines[x_idx].add(l_idx)
                
    blocked_count = [0] * (q*q)
    
    def get_blockers(x_idx):
        wl = valid_lines[x_idx]
        if len(wl) == 0:
            return set()
        if len(wl) == 1:
            l_idx = next(iter(wl))
            return set(lines_list[l_idx]) - {x_idx}
        else:
            l1 = next(iter(wl))
            l2 = next(iter(wl - {l1}))
            pts1 = set(lines_list[l1])
            pts2 = set(lines_list[l2])
            inter = pts1 & pts2
            if len(inter) == 1:
                y_idx = next(iter(inter))
                if y_idx != x_idx:
                    for l in wl:
                        if y_idx not in lines_list[l]:
                            return set()
                    return {y_idx}
            return set()
            
    for x_idx in range(q*q):
        for y_idx in get_blockers(x_idx):
            blocked_count[y_idx] += 1
            
    start_time = time.time()
    while True:
        if time.time() - start_time > 500:
            break
            
        added = False
        for y_idx in range(q*q):
            if y_idx in C:
                continue
            if blocked_count[y_idx] == 0:
                C.add(y_idx)
                added = True
                
                affected_x = []
                for x_idx in range(q*q):
                    if x_idx == y_idx:
                        continue
                    if valid_lines[x_idx] & lines_through_set[y_idx]:
                        affected_x.append(x_idx)
                        
                for x_idx in affected_x:
                    for y2_idx in get_blockers(x_idx):
                        blocked_count[y2_idx] -= 1
                        
                for x_idx in affected_x:
                    valid_lines[x_idx] -= lines_through_set[y_idx]
                    
                for x_idx in affected_x:
                    for y2_idx in get_blockers(x_idx):
                        blocked_count[y2_idx] += 1
                        
                break
                
        if not added:
            break
            
    N = [i for i in range(q*q) if i not in C]
    result = []
    for idx in N:
        pt = points[idx]
        x, y = pt
        result.append([[x[0], x[1]], [y[0], y[1]]])
        
    return np.array(result, dtype=np.int64)
