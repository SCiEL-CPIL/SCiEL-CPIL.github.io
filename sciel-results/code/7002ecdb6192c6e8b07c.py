"""
Minimum maximal no-isosceles set on [0,N)^2.

Strategy:
1. Precompute conflict bitsets: Conflicts[i][j] is a bitset (Python int) where bit k is 1
   if points i, j, k form an isosceles triangle.
2. Greedy Construction:
   - Maintain S (list) and S_bits (int).
   - Maintain C (list of candidate indices).
   - For each p in C, determine which q in C remain valid.
     q is valid if (Conflicts[p][q] & S_bits) == 0.
   - Pick p that minimizes the number of remaining valid q's.
   - Add p to S, update S_bits, update C.
3. Pruning:
   - Maintain BlockedMask: bitset of points that are blocked by the current set S.
     BlockedMask = OR_{u,v in S} Conflicts[u][v].
   - S is maximal if BlockedMask contains all points not in S.
   - Try removing points one by one. If S \ {p} is still maximal, keep it removed.
4. Multiple restarts with random tie-breaking to find the minimum size set.

Refinements from inspirations:
- Use vectorized NumPy operations for speedup in candidate evaluation.
- Add a "polish" phase after pruning: try to swap a point for a better one that
  maintains maximality but potentially allows further reduction.
- Use a deadline guard to ensure we return a valid result before timeout.
"""

import random
import time
import itertools
import numpy as np

N = 16
ALL_POINTS = list(itertools.product(range(N), repeat=2))
N_PTS = len(ALL_POINTS)

# Precompute squared distances
coords = np.array(ALL_POINTS)
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist2 = np.sum(diff**2, axis=2)

# Precompute Conflicts
# Conflicts[i][j] is an int where bit k is set if dist2[i,k] == dist2[j,k] or 
# dist2[i,k] == dist2[i,j] or dist2[j,k] == dist2[i,j]
# Note: We only care about distinct points.
Conflicts = [[0] * N_PTS for _ in range(N_PTS)]

def _precompute_conflicts():
    for i in range(N_PTS):
        for j in range(i + 1, N_PTS):
            mask = 0
            dij = dist2[i, j]
            for k in range(N_PTS):
                if k == i or k == j:
                    continue
                dik = dist2[i, k]
                djk = dist2[j, k]
                if dik == djk or dik == dij or djk == dij:
                    mask |= (1 << k)
            Conflicts[i][j] = mask
            Conflicts[j][i] = mask

_start_time = time.time()
_precompute_conflicts()

def _greedy_blocking_power(deadline):
    """
    Greedy algorithm that picks the point which reduces the candidate set the most.
    Uses bitsets for fast validity checks.
    """
    S = []
    S_bits = 0
    C = set(range(N_PTS))
    
    while C:
        if time.time() > deadline:
            break
        
        best_p = -1
        best_new_C_size = float('inf')
        best_new_C = None
        
        # Randomize order to break ties
        candidates = list(C)
        random.shuffle(candidates)
        
        for p in candidates:
            # Calculate new C if we add p
            # q is valid if (Conflicts[p][q] & S_bits) == 0
            new_C = set()
            for q in C:
                if q == p:
                    continue
                # Check if q is blocked by S given p
                if (Conflicts[p][q] & S_bits) == 0:
                    new_C.add(q)
            
            new_C_size = len(new_C)
            if new_C_size < best_new_C_size:
                best_new_C_size = new_C_size
                best_p = p
                best_new_C = new_C
        
        if best_p == -1:
            break
            
        S.append(best_p)
        S_bits |= (1 << best_p)
        C = best_new_C
        
    return S

def _is_maximal_fast(S):
    """
    Check if S is maximal using BlockedMask.
    S is maximal if every point x not in S is blocked by some pair in S.
    BlockedMask = OR_{u,v in S} Conflicts[u][v]
    """
    if not S:
        return False
        
    S_set = set(S)
    S_mask = 0
    for p in S:
        S_mask |= (1 << p)
        
    blocked_mask = 0
    # OR all conflicts for pairs in S
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            blocked_mask |= Conflicts[S[i]][S[j]]
            
    # Check if all points not in S are in blocked_mask
    # All points mask
    all_mask = (1 << N_PTS) - 1
    # Points not in S
    not_in_S_mask = all_mask & ~S_mask
    
    # If any point in not_in_S_mask is not in blocked_mask, it can be added
    if (not_in_S_mask & ~blocked_mask) != 0:
        return False
        
    return True

def _prune_maximal_set(S_list):
    """
    Try to reduce the size of a maximal isosceles-free set by removing points
    that do not compromise maximality.
    """
    S = list(S_list)
    changed = True
    while changed:
        changed = False
        # Iterate over a copy of the list
        for idx in range(len(S)):
            p = S[idx]
            S_temp = S[:idx] + S[idx+1:]
            
            # Check if S_temp is maximal
            if _is_maximal_fast(S_temp):
                S = S_temp
                changed = True
                break # Restart the loop with the new set
                
    return S

def _polish_maximal_set(S_list, deadline):
    """
    Polish a maximal set by trying to swap points for better ones.
    A swap is valid if the new set is still maximal and has the same size.
    This can sometimes enable further pruning.
    """
    S = list(S_list)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try each point in S
        for idx in range(len(S)):
            p = S[idx]
            S_without_p = S[:idx] + S[idx+1:]
            
            # Find all points that can replace p
            # A point q can replace p if S_without_p U {q} is valid and maximal
            # But we want to maintain the same size, so we just need S_without_p U {q} to be valid
            # and then check if it's maximal
            candidates = []
            for q in range(N_PTS):
                if q in S:
                    continue
                # Check if S_without_p U {q} is valid
                # This means for all u,v in S_without_p, {q,u,v} is not isosceles
                # and for all u in S_without_p, {p,u,q} is not isosceles? No, p is removed.
                # Actually, we need to check that q is compatible with all pairs in S_without_p
                valid = True
                for u in S_without_p:
                    for v in S_without_p:
                        if u == v:
                            continue
                        # Check if {q, u, v} is isosceles
                        # This is equivalent to checking if q is in Conflicts[u][v]
                        if (Conflicts[u][v] >> q) & 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    candidates.append(q)
            
            # Try each candidate
            for q in candidates:
                S_new = S_without_p + [q]
                if _is_maximal_fast(S_new):
                    S = S_new
                    improved = True
                    break
            if improved:
                break
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    best_set = None
    best_size = float('inf')
    
    # Multiple restarts
    while time.time() < deadline - 10.0:
        candidate = _greedy_blocking_power(deadline - 5.0)
        
        if candidate is None:
            continue
        
        # Prune the maximal set to reduce its size
        pruned_candidate = _prune_maximal_set(candidate)
        
        # Polish the set to try to find a better configuration
        polished_candidate = _polish_maximal_set(pruned_candidate, deadline - 5.0)
        
        if len(polished_candidate) < best_size:
            best_size = len(polished_candidate)
            best_set = polished_candidate
            
    if best_set is None:
        # Fallback: simple random greedy
        S = []
        S_bits = 0
        C = set(range(N_PTS))
        while C:
            p = random.choice(list(C))
            S.append(p)
            S_bits |= (1 << p)
            new_C = set()
            for q in C:
                if q == p:
                    continue
                if (Conflicts[p][q] & S_bits) == 0:
                    new_C.add(q)
            C = new_C
        best_set = _prune_maximal_set(S)

    return [ALL_POINTS[i] for i in best_set]
