import time
import random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[i][j] = _lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
                
    # Precompute conflict masks
    # conf_mask[a][b] is a bitmask of c such that {a,b,c} is bad
    conf_mask = [[0] * (N + 1) for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L = lcm_table[a][b]
            mask = 0
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if lcm_table[a][c] == L and lcm_table[b][c] == L:
                    mask |= (1 << c)
            conf_mask[a][b] = mask
            conf_mask[b][a] = mask
            
    best_set = []
    
    # Helper to calculate blocked mask for a given set
    def get_blocked_mask(S):
        blocked = 0
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                blocked |= conf_mask[S[i]][S[j]]
        return blocked

    # Greedy with bitmask
    def greedy_random():
        order = elements[:]
        random.shuffle(order)
        S = []
        blocked = 0
        for x in order:
            if not (blocked & (1 << x)):
                S.append(x)
                # Update blocked
                for y in S[:-1]: # S[:-1] is S before adding x
                    blocked |= conf_mask[x][y]
        return S

    # Run multiple restarts
    while time.time() < deadline - 10.0:
        S = greedy_random()
        if len(S) > len(best_set):
            best_set = S

    # Local Search
    # Try to improve best_set
    # 1. Try adding any x
    # 2. Try swapping: remove x, add y (if size same, try add z)
    
    S = best_set[:]
    blocked = get_blocked_mask(S)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add
        for x in range(1, N + 1):
            if not (blocked & (1 << x)):
                S.append(x)
                for y in S[:-1]:
                    blocked |= conf_mask[x][y]
                improved = True
                break
        
        if improved:
            continue
            
        # If no single addition, try 1-1 swap then addition?
        # Or just random perturbation
        if random.random() < 0.3:
            # Remove 1 random element
            if S:
                idx = random.randint(0, len(S) - 1)
                removed = S.pop(idx)
                # Recalculate blocked
                blocked = get_blocked_mask(S)
                
                # Try to add 2 elements?
                # Just try to add 1
                added_any = False
                for x in range(1, N + 1):
                    if x == removed:
                        continue
                    if not (blocked & (1 << x)):
                        S.append(x)
                        for y in S[:-1]:
                            blocked |= conf_mask[x][y]
                        added_any = True
                        break
                
                if not added_any:
                    # Put back
                    S.append(removed)
                    blocked = get_blocked_mask(S)
                elif len(S) <= len(best_set):
                    # If size didn't increase, revert to best_set?
                    # Actually, we want to keep the state that allows future growth.
                    # But for now, we only update best_set if size increases.
                    pass

        if len(S) > len(best_set):
            best_set = S[:]

    return best_set
