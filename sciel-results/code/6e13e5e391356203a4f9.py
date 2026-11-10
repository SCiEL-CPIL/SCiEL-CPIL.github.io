import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_PTS = N * N
FULL_MASK = (1 << NUM_PTS) - 1

# Precompute geometry
POINTS = [(x, y) for y in range(N) for x in range(N)]

# Precompute squared distances
dist = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for i in range(NUM_PTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_PTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        dist[i][j] = d
        dist[j][i] = d

# Precompute Iso masks: Iso[i][j] is bitmask of k such that triangle i,j,k is isosceles
Iso = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for i in range(NUM_PTS):
    for j in range(i + 1, NUM_PTS):
        d_ij = dist[i][j]
        mask = 0
        for k in range(NUM_PTS):
            if k == i or k == j:
                continue
            d_ik = dist[i][k]
            d_jk = dist[j][k]
            if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                mask |= (1 << k)
        Iso[i][j] = mask
        Iso[j][i] = mask

def greedy_maximal_set(order):
    """Build a maximal no-isosceles set by greedily adding points from order."""
    S = []
    blocked_mask = 0
    S_mask = 0
    for p in order:
        if (S_mask >> p) & 1:
            continue
        if (blocked_mask >> p) & 1:
            continue
        
        # Add p
        S.append(p)
        S_mask |= (1 << p)
        # Update blocked mask with new pairs involving p
        for u in S[:-1]:
            blocked_mask |= Iso[u][p]
            
    return S

def solve_min_maximal_k(K, deadline):
    """
    Try to find a maximal no-isosceles set of exactly size K.
    Returns the set if found, else None.
    """
    def backtrack(start_idx, current_set, blocked_mask, S_mask):
        if time.time() > deadline:
            return None
        
        current_size = len(current_set)
        remaining_needed = K - current_size
        
        if remaining_needed == 0:
            # Check maximality
            if (S_mask | blocked_mask) == FULL_MASK:
                return current_set[:]
            return None
        
        if remaining_needed < 0:
            return None
            
        # Pruning: Not enough points left to choose from
        if NUM_PTS - start_idx < remaining_needed:
            return None
            
        # Iterate through candidates
        for p in range(start_idx, NUM_PTS):
            if time.time() > deadline:
                return None
                
            # Check if p can be added
            if (S_mask >> p) & 1:
                continue
            if (blocked_mask >> p) & 1:
                continue
                
            # Compute new state
            new_S_mask = S_mask | (1 << p)
            new_blocked = blocked_mask
            for u in current_set:
                new_blocked |= Iso[u][p]
                
            # Recurse
            res = backtrack(p + 1, current_set + [p], new_blocked, new_S_mask)
            if res is not None:
                return res
                
        return None

    return backtrack(0, [], 0, 0)

def construct():
    DEADLINE = time.time() + 290
    best_set = None
    best_len = float('inf')
    
    # Phase 1: Get an upper bound using Greedy
    rng = random.Random(42)
    for i in range(100):
        if time.time() > DEADLINE:
            break
        order = list(range(NUM_PTS))
        rng.shuffle(order)
        S = greedy_maximal_set(order)
        if len(S) < best_len:
            best_len = len(S)
            best_set = S.copy()
            
    if best_set is None:
        return [POINTS[i] for i in greedy_maximal_set(list(range(NUM_PTS)))]
        
    # Phase 2: B&B to find smaller maximal sets
    for K in range(1, best_len):
        if time.time() > DEADLINE:
            break
            
        found_set = solve_min_maximal_k(K, DEADLINE)
        if found_set is not None:
            best_set = found_set
            best_len = K
            if K == 1:
                break
                
    return [POINTS[i] for i in best_set]
