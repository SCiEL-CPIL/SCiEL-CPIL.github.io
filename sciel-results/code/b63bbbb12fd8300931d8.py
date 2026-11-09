import numpy as np
import time
import heapq


def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    k = 3
    
    n_points = q * q  # 2401
    
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 0), (0, 1)))
    n_dirs = len(directions)  # 50
    
    def f_mul(a, b):
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
        
    def f_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    field_elements = []
    for a in range(p):
        for b in range(p):
            field_elements.append((a, b))
            
    def get_coords(i):
        xa = i // 343
        xb = (i // 49) % 7
        ya = (i // 7) % 7
        yb = i % 7
        return xa, xb, ya, yb
        
    def get_index(xa, xb, ya, yb):
        return xa * 343 + xb * 49 + ya * 7 + yb

    line_members = [[None for _ in range(n_dirs)] for _ in range(n_points)]
    
    for i in range(n_points):
        xa, xb, ya, yb = get_coords(i)
        pt = ((xa, xb), (ya, yb))
        for d in range(n_dirs):
            v = directions[d]
            members = []
            for t_idx in range(1, q):
                t = field_elements[t_idx]
                tv_x = f_mul(t, v[0])
                tv_y = f_mul(t, v[1])
                new_x = f_add(pt[0], tv_x)
                new_y = f_add(pt[1], tv_y)
                nxa, nxb = new_x
                nya, nyb = new_y
                j = get_index(nxa, nxb, nya, nyb)
                members.append(j)
            line_members[i][d] = members
            
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
            
    S = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - 3 * yb * yb) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        idx = get_index(xa, xb, ya, yb)
                        S.add(idx)
                        
    valid_dirs = np.zeros(n_points, dtype=np.int32)
    is_valid = np.ones((n_points, n_dirs), dtype=bool)
    
    for x in range(n_points):
        for d in range(n_dirs):
            for y in line_members[x][d]:
                if y in S:
                    is_valid[x][d] = False
                    break
            if is_valid[x][d]:
                valid_dirs[x] += 1
                
    start_time = time.time()
    deadline = start_time + 590
    
    def compute_cost(p):
        if p in S:
            return 0
        cost = 0
        for d in range(n_dirs):
            for x in line_members[p][d]:
                if valid_dirs[x] == 1 and is_valid[x][d]:
                    cost += 1
        return cost
    
    # Phase 1: Priority queue greedy
    heap = []
    for p in range(n_points):
        if p not in S:
            c = compute_cost(p)
            heapq.heappush(heap, (c, p))
    
    while heap:
        if time.time() > deadline:
            break
        
        c, p = heapq.heappop(heap)
        
        if p in S:
            continue
        
        actual_cost = compute_cost(p)
        if actual_cost != c:
            heapq.heappush(heap, (actual_cost, p))
            continue
        
        if actual_cost > 0:
            continue
        
        S.add(p)
        for d in range(n_dirs):
            for x in line_members[p][d]:
                if is_valid[x][d]:
                    is_valid[x][d] = False
                    valid_dirs[x] -= 1
    
    # Phase 2: Additional passes to catch points that became removable
    for extra_pass in range(3):
        if time.time() > deadline:
            break
        
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for p in range(n_points):
                if p in S:
                    continue
                if compute_cost(p) == 0:
                    S.add(p)
                    changed = True
                    for d in range(n_dirs):
                        for x in line_members[p][d]:
                            if is_valid[x][d]:
                                is_valid[x][d] = False
                                valid_dirs[x] -= 1
    
    N_indices = [i for i in range(n_points) if i not in S]
    
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for idx, i in enumerate(N_indices):
        xa, xb, ya, yb = get_coords(i)
        result[idx, 0, 0] = xa
        result[idx, 0, 1] = xb
        result[idx, 1, 0] = ya
        result[idx, 1, 1] = yb
        
    return result
