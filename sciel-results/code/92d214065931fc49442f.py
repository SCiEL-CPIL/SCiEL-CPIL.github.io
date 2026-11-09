import time
import numpy as np

def construct():
    t_start = time.time()
    DEADLINE = t_start + 75.0
    N = 4096
    MAX_SIZE = 80
    
    selected = []
    triple_sums = set()
    
    for x in range(1, N + 1):
        if len(selected) >= MAX_SIZE:
            break
        if time.time() > DEADLINE - 5:
            break
        
        k = len(selected)
        
        if k == 0:
            selected.append(x)
            triple_sums.add(3 * x)
            continue
        
        sel = np.array(selected, dtype=np.int64)
        
        # Build all new triple sums involving x
        # (x,x,x): 1 sum
        # (x,x,y): k sums  
        # (x,y,z) with y<=z: k*(k+1)/2 sums
        total_new = 1 + k + k * (k + 1) // 2
        
        new_sums = np.empty(total_new, dtype=np.int64)
        new_sums[0] = 3 * x
        new_sums[1:1+k] = 2 * x + sel
        
        # (x, y, z) for 0<=i<=j<k
        if k > 1:
            yz = sel[:, None] + sel[None, :]
            mask = np.triu(np.ones((k, k), dtype=bool))
            yz_upper = yz[mask]
            new_sums[1+k:1+k+len(yz_upper)] = x + yz_upper
        elif k == 1:
            new_sums[1+k] = x + sel[0] + sel[0]
        
        # Check uniqueness within new_sums
        new_list = new_sums.tolist()
        if len(new_list) != len(set(new_list)):
            continue
        
        # Check against existing
        new_set = set(new_list)
        if new_set & triple_sums:
            continue
        
        # Accept
        triple_sums.update(new_set)
        selected.append(x)
    
    return selected
