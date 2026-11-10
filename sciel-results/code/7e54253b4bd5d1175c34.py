import numpy as np
import time
import random
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 570
    
    p = 7
    k = 3
    
    def mul(a, b):
        a1, a2 = a
        b1, b2 = b
        return ((a1*b1 + k*a2*b2) % p, (a1*b2 + a2*b1) % p)
    
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p)
                  for ya in range(p) for yb in range(p)]
    n = len(all_points)
    
    # Build lines
    directions = []
    vert_dict = defaultdict(list)
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        vert_dict[(xa, xb)].append(i)
    directions.append(list(vert_dict.values()))
    
    for ma in range(p):
        for mb in range(p):
            lines_dict = defaultdict(list)
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul((xa, xb), (ma, mb))
                c = ((ya - mx[0]) % p, (yb - mx[1]) % p)
                lines_dict[c].append(i)
            directions.append(list(lines_dict.values()))
    
    num_dirs = len(directions)
    
    # Precompute line membership and line id
    line_membership = []
    line_id = []
    for d in range(num_dirs):
        lm = np.array(directions[d], dtype=np.int32)
        lid = np.zeros(n, dtype=np.int32)
        for li in range(lm.shape[0]):
            for pt in lm[li]:
                lid[pt] = li
        line_membership.append(lm)
        line_id.append(lid)
    
    def check_validity(C):
        C_arr = np.zeros(n, dtype=bool)
        for c in C:
            C_arr[c] = True
        
        safe = np.zeros(n, dtype=bool)
        for d in range(num_dirs):
            lm = line_membership[d]
            line_c = C_arr[lm]
            counts = line_c.sum(axis=1)
            
            # Vectorized: for each point, check if safe in direction d
            for pt in range(n):
                if safe[pt]:
                    continue
                li = line_id[d][pt]
                cnt = counts[li]
                if cnt == 0 or (cnt == 1 and C_arr[pt]):
                    safe[pt] = True
        
        return safe.all()
    
    def in_unital(xa, xb, ya, yb):
        return (2 * xa + ya * ya - k * yb * yb) % p == 0
    
    unital_set = set()
    for i, (xa, xb, ya, yb) in enumerate(all_points):
        if in_unital(xa, xb, ya, yb):
            unital_set.add(i)
    
    qr = set((u * u) % p for u in range(p))
    
    best_C = None
    best_size = 0
    rng = random.Random(42)
    
    initial_candidates = [
        unital_set,
        {i for i in unital_set if all_points[i][2] in qr},
        {i for i in unital_set if all_points[i][2] not in qr},
    ]
    
    for init_C in initial_candidates:
        if time.time() > deadline - 20:
            break
        C = set(init_C)
        
        # Local search: remove points
        for _ in range(120):
            if time.time() > deadline - 20:
                break
            if not C:
                break
            x = rng.choice(list(C))
            C.remove(x)
            if check_validity(C):
                pass
            else:
                C.add(x)
        
        # Add points
        for _ in range(60):
            if time.time() > deadline - 20:
                break
            x = rng.randint(0, n - 1)
            if x in C:
                continue
            C.add(x)
            if check_validity(C):
                pass
            else:
                C.remove(x)
        
        if len(C) > best_size:
            best_size = len(C)
            best_C = C
    
    if best_C is None:
        best_C = set()
    
    N_indices = [i for i in range(n) if i not in best_C]
    pts = [all_points[i] for i in N_indices]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result
