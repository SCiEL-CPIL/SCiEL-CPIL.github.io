import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 16
    k = 4
    
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
    
    total = len(all_masks)
    chosen = []
    
    for i in range(total):
        if time.time() > deadline:
            break
        s = all_masks[i]
        
        compatible = True
        m = len(chosen)
        for ai in range(m):
            a = chosen[ai]
            a_s = a & s
            for bi in range(ai + 1, m):
                b = chosen[bi]
                if (a & b) == a_s and (b & s) == a_s:
                    compatible = False
                    break
            if not compatible:
                break
        
        if compatible:
            chosen.append(s)
    
    result = []
    for mask in chosen:
        elems = []
        for x in range(n):
            if mask & (1 << x):
                elems.append(x + 1)
        result.append(elems)
    
    return result
# EVOLVE-BLOCK-END
