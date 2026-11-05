import random
import time
import itertools
import math
import numpy as np

N = 16
ALL_POINTS = list(itertools.product(range(N), repeat=2))
N_PTS = len(ALL_POINTS)

# Precompute squared distances
coords = np.array(ALL_POINTS)
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist2 = np.sum(diff**2, axis=2)

# Precompute bad pairs for each point
# BAD_PAIRS[p] is a set of tuples (u, v) with u < v such that {p, u, v} is isosceles.
BAD_PAIRS = [set() for _ in range(N_PTS)]

def _precompute_bad_pairs():
    i_idx, j_idx = np.triu_indices(N_PTS, k=1)
    for p in range(N_PTS):
        dp = dist2[p, :]
        dp_i = dp[i_idx]
        dp_j = dp[j_idx]
        dij = dist2[i_idx, j_idx]
        cond = (dp_i == dp_j) | (dp_i == dij) | (dp_j == dij)
        bad_indices = np.where(cond)[0]
        for k in bad_indices:
            u, v = i_idx[k], j_idx[k]
            if u > v:
                u, v = v, u
            BAD_PAIRS[p].add((u, v))

_start_time = time.time()
_precompute_bad_pairs()

def _can_add(p, S_set):
    """Check if p can be added to S_set (represented as a set of indices)."""
    for u, v in BAD_PAIRS[p]:
        if u in S_set and v in S_set:
            return False
    return True

def _is_maximal(S_set):
    """Check if S_set is maximal."""
    for p in range(N_PTS):
        if p not in S_set:
            if _can_add(p, S_set):
                return False
    return True

def _greedy_set_cover(deadline):
    """
    Greedy construction using Set Cover heuristic.
    Prioritizes points that block the most other points.
    """
    S = []
    S_set = set()
    C = set(range(N_PTS))
    
    while C:
        if time.time() > deadline:
            break
        
        best_p = -1
        best_coverage = -1
        best_new_C = None
        
        candidates = list(C)
        random.shuffle(candidates)
        
        for p in candidates:
            coverage = 0
            new_C = set()
            
            for q in C:
                if q == p:
                    continue
                # Check if q is valid with S U {p}
                # q is valid with S.
                # Check if q conflicts with p given S.
                # Conflict if exists u in S such that {p, u, q} is isosceles.
                valid = True
                for u in S:
                    a, b = (p, u) if p < u else (u, p)
                    if (a, b) in BAD_PAIRS[q]:
                        valid = False
                        break
                
                if valid:
                    new_C.add(q)
                else:
                    coverage += 1
                    
            if coverage > best_coverage:
                best_coverage = coverage
                best_p = p
                best_new_C = new_C
        
        if best_p == -1:
            break
            
        S.append(best_p)
        S_set.add(best_p)
        C = best_new_C
        
    return S

def _local_search(S_initial, deadline):
    """
    Local search on Maximal Sets.
    Uses swaps to explore the space of maximal sets of the same size.
    """
    S_set = set(S_initial)
    
    # Verify initial set is maximal
    if not _is_maximal(S_set):
        # Repair
        while True:
            addable = [p for p in range(N_PTS) if p not in S_set and _can_add(p, S_set)]
            if not addable:
                break
            p = random.choice(addable)
            S_set.add(p)
            
    best_set = set(S_set)
    best_size = len(S_set)
    
    while time.time() < deadline:
        if not S_set:
            break
            
        u = random.choice(list(S_set))
        v = random.choice(list(set(range(N_PTS)) - S_set))
        
        S_minus_u = S_set - {u}
        S_new = S_minus_u | {v}
        
        # Check if v is compatible with S_minus_u
        valid = True
        for w, x in BAD_PAIRS[v]:
            if w in S_minus_u and x in S_minus_u:
                valid = False
                break
                
        if not valid:
            continue
            
        # Check if S_new is Maximal
        if _is_maximal(S_new):
            S_set = S_new
            # Since size is same, we don't update best_size here directly
            # but we keep exploring.
            
    return list(best_set)

def construct():
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin
    
    best_set = None
    best_size = float('inf')
    
    # Multiple restarts with Set Cover Greedy
    restart_count = 0
    while time.time() < deadline - 10.0:
        restart_count += 1
        
        S_init = _greedy_set_cover(deadline - 10.0)
        
        if S_init:
            # Local search to explore the basin
            S_opt = _local_search(S_init, deadline - 5.0)
            
            if len(S_opt) < best_size:
                best_size = len(S_opt)
                best_set = S_opt
                
    if best_set is None:
        # Fallback
        best_set = _greedy_set_cover(deadline)
        
    return [ALL_POINTS[i] for i in best_set]
