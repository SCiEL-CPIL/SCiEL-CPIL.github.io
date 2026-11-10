import numpy as np
from itertools import combinations, product
import time

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600
    
    d = 11
    
    # Step 1: Find supports (5-subsets) with intersection <= 2
    supports = []
    all_subsets = list(combinations(range(d), 5))
    
    # Greedy packing
    for s in all_subsets:
        if time.time() > deadline - 10:
            break
        valid = True
        for t in supports:
            if len(set(s) & set(t)) > 2:
                valid = False
                break
        if valid:
            supports.append(s)
            
    # Step 2: For each support, generate sign patterns
    # Need binary code length 5, min dist 2.
    # Even parity code has size 16.
    sign_patterns = []
    for bits in product([1, -1], repeat=5):
        prod = 1
        for b in bits:
            prod *= b
        if prod == 1: # Even number of -1s? No, product 1 means even number of -1s?
             # 1*-1*-1 = 1. Yes.
             # This is not a code with min dist 2 necessarily?
             # Distance between two even parity vectors is even.
             # Min non-zero distance is 2.
             # So this works.
             sign_patterns.append(bits)
             
    # Actually, any two vectors in the even parity code have dot product:
    # u, v in {1, -1}^5. u*v = 1.
    # u.v = sum u_i v_i.
    # If u=v, dot=5.
    # If u!=v, dist >= 2.
    # If dist=2, 2 positions differ.
    # u_i v_i = -1 for 2 pos, 1 for 3 pos.
    # Dot = 3 - 2 = 1.
    # Condition u.v <= 2. 1 <= 2. OK.
    
    points = []
    for s in supports:
        for p in sign_patterns:
            v = np.zeros(d, dtype=np.int64)
            for i, idx in enumerate(s):
                v[idx] = p[i]
            points.append(v)
            
    # Convert to array
    if not points:
         # Fallback to R=2 solution
         points = []
         for i in range(d):
             for j in range(i+1, d):
                 for si in [1, -1]:
                     for sj in [1, -1]:
                         v = np.zeros(d, dtype=np.int64)
                         v[i] = si
                         v[j] = sj
                         points.append(v)
                         
    points_arr = np.array(points, dtype=np.int64)
    
    # Verification
    # Check max norm
    norms = np.sum(points_arr**2, axis=1)
    max_norm = np.max(norms)
    
    # Check min dist
    min_dist = float('inf')
    k = len(points_arr)
    for i in range(k):
        for j in range(i+1, k):
            diff = points_arr[i] - points_arr[j]
            dist_sq = np.sum(diff**2)
            if dist_sq < min_dist:
                min_dist = dist_sq
    if min_dist < max_norm:
        # Invalid, fallback to R=2
        points_arr = np.array([np.ones(d), -1*np.ones(d)], dtype=np.int64)
        
    return points_arr
