import random
import time
from itertools import combinations
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    N = 128
    # Precompute LCM table for speed
    lcm_table = [[1] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[i][j] = lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
            
    # Precompute forbidden pairs for each x
    # forb_pairs[x] is a list of (a, b) such that {a, b, x} is a forbidden triple
    forb_pairs = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
                    # {a, b, c} is forbidden
                    forb_pairs[a].append((b, c))
                    forb_pairs[b].append((a, c))
                    forb_pairs[c].append((a, b))
                    
    def can_add(x, S):
        # Check if x can be added to S
        # S is a set or a boolean list? 
        # Using a boolean list for O(1) access is faster
        pass

    def greedy_random():
        S = [False] * (N + 1)
        C = list(range(1, N + 1))
        random.shuffle(C)
        
        size = 0
        while C:
            valid_candidates = []
            # Find all x in C that can be added
            for x in C:
                ok = True
                for a, b in forb_pairs[x]:
                    if S[a] and S[b]:
                        ok = False
                        break
                if ok:
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # Pick a random valid candidate
            # Heuristic: Pick the one that blocks the fewest others? 
            # For now, random to explore different landscapes.
            x = random.choice(valid_candidates)
            S[x] = True
            C.remove(x)
            size += 1
            
        return S, size

    best_S = None
    best_size = 0
    
    # Initial deterministic greedy (ascending)
    S_det = [False] * (N + 1)
    C_det = list(range(1, N + 1))
    size_det = 0
    while C_det:
        valid = []
        for x in C_det:
            ok = True
            for a, b in forb_pairs[x]:
                if S_det[a] and S_det[b]:
                    ok = False
                    break
            if ok:
                valid.append(x)
        if not valid:
            break
        x = valid[0] # Smallest
        S_det[x] = True
        C_det.remove(x)
        size_det += 1
        
    if size_det > best_size:
        best_size = size_det
        best_S = S_det[:]

    # Randomized Greedy Restarts
    iterations = 0
    while time.time() < deadline:
        S, size = greedy_random()
        if size > best_size:
            best_size = size
            best_S = S[:]
        iterations += 1
        
    # Local Search on best_S
    # Try to swap or add
    # If we can add, we would have done it in greedy.
    # So we are likely stuck.
    # Try removing one element and re-greedy-ing?
    if best_S and time.time() < deadline - 5:
        current_S = best_S[:]
        current_size = best_size
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing each element u in current_S
            for u in range(1, N + 1):
                if not current_S[u]:
                    continue
                # Remove u
                current_S[u] = False
                temp_size = current_size - 1
                temp_S = current_S[:]
                temp_C = [x for x in range(1, N + 1) if not temp_S[x]]
                
                # Greedy add
                while temp_C:
                    valid = []
                    for x in temp_C:
                        ok = True
                        for a, b in forb_pairs[x]:
                            if temp_S[a] and temp_S[b]:
                                ok = False
                                break
                        if ok:
                            valid.append(x)
                    if not valid:
                        break
                    # Pick random or smallest? Random for diversity
                    x = random.choice(valid)
                    temp_S[x] = True
                    temp_C.remove(x)
                    temp_size += 1
                    
                if temp_size > current_size:
                    current_S = temp_S
                    current_size = temp_size
                    best_size = current_size
                    best_S = current_S[:]
                    improved = True
                    break # Restart loop with new best
            
    # Convert best_S to list of elements
    result = [i for i in range(1, N + 1) if best_S[i]]
    return result
