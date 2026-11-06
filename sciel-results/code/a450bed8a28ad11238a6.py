import time
import random
from itertools import combinations

N = 16
K = 4
ALL_COMBOS = list(combinations(range(1, N + 1), K))
M = len(ALL_COMBOS)

ALL_SETS = [0] * M
for i, c in enumerate(ALL_COMBOS):
    mask = 0
    for x in c:
        mask |= (1 << (x - 1))
    ALL_SETS[i] = mask

INTER = [[0]*M for _ in range(M)]
for i in range(M):
    mi = ALL_SETS[i]
    for j in range(i+1, M):
        inter_val = mi & mj
        INTER[i][j] = inter_val
        INTER[j][i] = inter_val

def count_conflicts(x_idx, family_indices):
    n = len(family_indices)
    count = 0
    inter_x = INTER[x_idx]
    inter = INTER
    for u in range(n):
        i = family_indices[u]
        xi = inter_x[i]
        for v in range(u + 1, n):
            j = family_indices[v]
            if inter_x[j] == xi and inter[i][j] == xi:
                count += 1
    return count

def count_conflicts_with_a(x_idx, a_idx, family_indices):
    # Count C in family_indices (C != x_idx, C != a_idx)
    # such that x_idx ^ a_idx ^ C is a sunflower? No.
    # Condition: x_idx ∩ a_idx == x_idx ∩ C == a_idx ∩ C
    s_xa = INTER[x_idx][a_idx]
    count = 0
    inter_x = INTER[x_idx]
    inter_a = INTER[a_idx]
    for i in family_indices:
        if i == x_idx or i == a_idx:
            continue
        if inter_x[i] == s_xa and inter_a[i] == s_xa:
            count += 1
    return count

def construct():
    start_time = time.time()
    timeout = 50.0
    
    best_family = []
    best_size = 0
    
    # Greedy Init
    random.seed(0)
    indices = list(range(M))
    random.shuffle(indices)
    
    current_family = []
    in_family = [False] * M
    
    for idx in indices:
        if count_conflicts(idx, current_family) == 0:
            current_family.append(idx)
            in_family[idx] = True
            
    best_size = len(current_family)
    best_family = current_family[:]
    
    # Local Search
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() - start_time < timeout:
        if no_improve > max_no_improve:
            # Kick
            if len(current_family) > 10:
                kick_size = min(5, len(current_family) // 5)
                for _ in range(kick_size):
                    if current_family:
                        rem = random.choice(current_family)
                        current_family.remove(rem)
                        in_family[rem] = False
            no_improve = 0
            max_no_improve += 500

        # Try Swap
        if current_family:
            a_idx = random.choice(current_family)
            # Pick random B
            b_idx = random.randint(0, M - 1)
            if in_family[b_idx]:
                continue
                
            # Check swap validity
            # New conflicts = count_conflicts(b_idx, current_family) - count_conflicts_with_a(b_idx, a_idx, current_family)
            c_total = count_conflicts(b_idx, current_family)
            c_with_a = count_conflicts_with_a(b_idx, a_idx, current_family)
            
            if c_total - c_with_a == 0:
                # Accept swap
                current_family.remove(a_idx)
                in_family[a_idx] = False
                current_family.append(b_idx)
                in_family[b_idx] = True
                no_improve = 0
                
                if len(current_family) > best_size:
                    best_size = len(current_family)
                    best_family = current_family[:]
            else:
                no_improve += 1
        else:
            # If empty, just add
            idx = random.randint(0, M - 1)
            if not in_family[idx]:
                if count_conflicts(idx, current_family) == 0:
                    current_family.append(idx)
                    in_family[idx] = True
                    no_improve = 0
                    if len(current_family) > best_size:
                        best_size = len(current_family)
                        best_family = current_family[:]

    result = []
    for idx in best_family:
        mask = ALL_SETS[idx]
        elems = []
        for i in range(N):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(elems)
        
    return result
