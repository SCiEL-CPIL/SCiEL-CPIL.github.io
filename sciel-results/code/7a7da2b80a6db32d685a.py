import itertools
import random
import time

# Precompute all 4-subsets of [1, 16]
SUBSETS = []
MASKS = []
for c in itertools.combinations(range(16), 4):
    m = 0
    for x in c:
        m |= (1 << x)
    MASKS.append(m)
    SUBSETS.append(c)

N = len(SUBSETS)

# Precompute intersection sizes
# inter[i][j] = |S_i \cap S_j|
inter = [[0] * N for _ in range(N)]
for i in range(N):
    mi = MASKS[i]
    for j in range(i, N):
        mj = MASKS[j]
        try:
            cnt = (mi & mj).bit_count()
        except AttributeError:
            cnt = bin(mi & mj).count('1')
        inter[i][j] = cnt
        inter[j][i] = cnt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for 90s limit
    
    best_family = []
    
    def greedy_pass(order):
        family = []
        inter_local = inter
        family_local = family
        
        for k in order:
            valid = True
            row_k = inter_local[k]
            
            # Check against all pairs in family
            for idx_i in range(len(family_local)):
                i = family_local[idx_i]
                ik = row_k[i]
                for idx_j in range(idx_i + 1, len(family_local)):
                    j = family_local[idx_j]
                    # inter[i][j] == ik and inter[j][k] == ik
                    # inter[j][k] is row_k[j]
                    if inter_local[i][j] == ik and row_k[j] == ik:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                family_local.append(k)
                
        return family_local

    # Initial deterministic pass
    best_family = greedy_pass(list(range(N)))
    
    # Random restarts
    while time.time() < deadline:
        order = list(range(N))
        random.shuffle(order)
        
        current_family = greedy_pass(order)
        
        if len(current_family) > len(best_family):
            best_family = current_family
            
        if time.time() > deadline - 2:
            break
            
    result = []
    for idx in best_family:
        s = tuple(x + 1 for x in SUBSETS[idx])
        result.append(sorted(s))
        
    return result
