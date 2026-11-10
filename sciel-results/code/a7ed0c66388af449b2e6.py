import numpy as np
import random
import time
import itertools

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precompute distance matrix
D = np.zeros((NUM_POINTS, NUM_POINTS), dtype=np.int32)
for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        dx = POINTS[i][0] - POINTS[j][0]
        dy = POINTS[i][1] - POINTS[j][1]
        d = dx * dx + dy * dy
        D[i, j] = d
        D[j, i] = d

def is_isosceles_idx(v, a, b):
    d_va = D[v, a]
    d_vb = D[v, b]
    d_ab = D[a, b]
    return d_va == d_vb or d_va == d_ab or d_vb == d_ab

def is_maximal(S):
    S_list = list(S)
    S_set = set(S_list)
    # Check all points not in S
    for v in range(NUM_POINTS):
        if v in S_set:
            continue
        blocked = False
        # Check pairs in S
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                if is_isosceles_idx(v, a, b):
                    blocked = True
                    break
            if blocked:
                break
        if not blocked:
            return False
    return True

def construct():
    best_set = None
    best_size = float('inf')
    deadline = time.time() + 290  # Safety margin
    
    # Initial greedy to get a baseline
    random.seed(42)
    for _ in range(10):
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        S = []
        S_set = set()
        for p in order:
            if all(not is_isosceles_idx(p, a, b) for a, b in itertools.combinations(S, 2)):
                S.append(p)
                S_set.add(p)
        
        # Local search: try to remove points
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle to try different removals
            S_copy = S[:]
            random.shuffle(S_copy)
            for p in S_copy:
                if time.time() > deadline:
                    break
                S_candidate = [x for x in S if x != p]
                # Quick check: is p blocked by S_candidate?
                # If not, S_candidate is not maximal (p can be added)
                p_blocked = False
                for a, b in itertools.combinations(S_candidate, 2):
                    if is_isosceles_idx(p, a, b):
                        p_blocked = True
                        break
                
                if not p_blocked:
                    continue
                
                # Full check: is S_candidate maximal?
                if is_maximal(S_candidate):
                    S = S_candidate
                    improved = True
                    break # Restart loop with new S
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            
        if time.time() > deadline:
            break

    if best_set is None:
        # Fallback: simple greedy
        S = []
        for p in range(NUM_POINTS):
            if all(not is_isosceles_idx(p, a, b) for a, b in itertools.combinations(S, 2)):
                S.append(p)
        best_set = S

    # Convert indices back to (x, y) tuples
    return [POINTS[i] for i in best_set]
