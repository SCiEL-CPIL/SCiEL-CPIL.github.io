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

def compute_blocked_mask(S_list):
    """Return the union of Iso masks for all pairs in S_list."""
    mask = 0
    n = len(S_list)
    for i in range(n):
        pi = S_list[i]
        for j in range(i + 1, n):
            pj = S_list[j]
            mask |= Iso[pi][pj]
    return mask

def is_valid_mask(S_list, S_mask, blocked_mask):
    """Check if no three points in S_list form an isosceles triangle."""
    return (S_mask & blocked_mask) == 0

def is_maximal_mask(S_mask, blocked_mask):
    """Check if no point outside S_list can be added."""
    return (S_mask | blocked_mask) == FULL_MASK

def greedy_maximal_set(order):
    """Build a maximal no-isosceles set by greedily adding points from order.
    Uses running blocked_mask for O(n*|S|) complexity."""
    S = []
    blocked_mask = 0
    for p in order:
        if (blocked_mask >> p) & 1:
            continue
        # Check if p conflicts with existing pairs
        # p is blocked if it's in Iso[i][j] for any i,j in S
        # This is exactly what blocked_mask represents.
        # So if (blocked_mask >> p) & 1 is 0, p is safe to add.
        S.append(p)
        for i in range(len(S) - 1):
            blocked_mask |= Iso[S[i]][p]
    return S

def local_search(S, deadline):
    """Local search: try removals and swaps to minimize the maximal set.
    Optimized with bitmasks."""
    best_S = S[:]
    best_len = len(best_S)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        n = len(best_S)
        
        # Try removing each point
        for i in range(n):
            if time.time() > deadline:
                break
            
            # Create S without best_S[i]
            S_trial = best_S[:i] + best_S[i+1:]
            S_trial_mask = 0
            for p in S_trial:
                S_trial_mask |= (1 << p)
            
            blocked_trial = compute_blocked_mask(S_trial)
            
            # Validity is guaranteed if parent was valid (subset of valid set is valid)
            # Check maximality
            if is_maximal_mask(S_trial_mask, blocked_trial):
                best_S = S_trial
                best_len = n - 1
                improved = True
                break
        
        if not improved:
            # Try swaps: Remove p, add q
            # We iterate p in S, then q not in S
            for i in range(len(best_S)):
                if time.time() > deadline:
                    break
                
                p = best_S[i]
                S_without_p = best_S[:i] + best_S[i+1:]
                S_without_p_mask = 0
                for q in S_without_p:
                    S_without_p_mask |= (1 << q)
                
                blocked_without_p = compute_blocked_mask(S_without_p)
                
                # Candidates for q:
                # 1. Not in S_without_p
                # 2. Not blocked by S_without_p (i.e., not in blocked_without_p)
                #    If q is in blocked_without_p, it forms isosceles with some pair in S_without_p.
                # 3. Adding q must result in a maximal set.
                
                # Iterate through all possible q
                # Optimization: We can iterate through bits of ~S_without_p_mask & ~blocked_without_p
                # But we also need to check if the NEW set is maximal.
                
                candidates_mask = FULL_MASK & ~S_without_p_mask & ~blocked_without_p
                # Note: q must also not form isosceles with pairs in S_without_p.
                # Wait, blocked_without_p IS the set of points that form isosceles with pairs in S_without_p.
                # So any q in candidates_mask is valid to add to S_without_p.
                
                # Now we check if S_without_p U {q} is maximal.
                # If we add q, new_blocked = blocked_without_p | (union of Iso[q][r] for r in S_without_p)
                # new_mask = S_without_p_mask | (1 << q)
                # Maximal if new_mask | new_blocked == FULL_MASK
                
                # Iterate candidates
                # To avoid iterating all 256 points, we can just iterate through set bits
                # But 256 is small.
                for q in range(NUM_PTS):
                    if time.time() > deadline:
                        break
                    if not (candidates_mask >> q) & 1:
                        continue
                    
                    # Compute new blocked mask
                    new_blocked = blocked_without_p
                    for r in S_without_p:
                        new_blocked |= Iso[q][r]
                    
                    new_mask = S_without_p_mask | (1 << q)
                    
                    if is_maximal_mask(new_mask, new_blocked):
                        best_S = S_without_p + [q]
                        improved = True
                        break
                if improved:
                    break
    
    return best_S

def construct():
    DEADLINE = time.time() + 290
    best_set = None
    best_len = float('inf')
    
    # Phase 1: Greedy restarts
    rng = random.Random(0)
    # Increase restarts if greedy is fast
    for i in range(2000):
        if time.time() > DEADLINE:
            break
        order = list(range(NUM_PTS))
        rng.shuffle(order)
        S = greedy_maximal_set(order)
        if len(S) < best_len:
            best_len = len(S)
            best_set = S.copy()
            # Optional: break early if we find a very small set?
            # Minimal is likely around 10-15.
            if best_len <= 10:
                break

    if best_set is None:
        return [POINTS[i] for i in greedy_maximal_set(list(range(NUM_PTS)))]
    
    # Phase 2: Local search on best greedy result
    S = local_search(best_set, DEADLINE)
    if len(S) < best_len:
        best_len = len(S)
        best_set = S
    
    # Phase 3: More greedy + local search on diverse starts
    # Focus on starts that produced small sets
    for i in range(500):
        if time.time() > DEADLINE:
            break
        order = list(range(NUM_PTS))
        rng.shuffle(order)
        S_greedy = greedy_maximal_set(order)
        if len(S_greedy) < best_len + 2:
            S_ls = local_search(S_greedy, DEADLINE)
            if len(S_ls) < best_len:
                best_len = len(S_ls)
                best_set = S_ls
    
    return [POINTS[i] for i in best_set]
