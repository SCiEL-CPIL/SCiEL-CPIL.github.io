"""
Minimum maximal no-isosceles set on [0,N)^2.

Strategy: Optimized Greedy with Precomputed Coverage.
1. Precompute Q[p][u] = set of q such that {p, u, q} is isosceles.
2. Greedy: Pick p in C that minimizes |C \ (Union_{u in S} Q[p][u])|.
3. Local Search: Try to swap points to reduce size.
"""

import random
import time
import itertools
import numpy as np
from collections import defaultdict

N = 16
ALL_POINTS = list(itertools.product(range(N), repeat=2))
N_PTS = len(ALL_POINTS)

# Precompute squared distances
coords = np.array(ALL_POINTS)
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist2 = np.sum(diff**2, axis=2)

# Q[p][u] is a set of q indices
# If {p, u, q} is isosceles, q is in Q[p][u]
# Note: Q is symmetric in p and u, so Q[p][u] == Q[u][p]
Q = [dict() for _ in range(N_PTS)]

def _precompute_coverage():
    """
    Precompute the coverage sets Q[p][u].
    """
    # Initialize Q
    for p in range(N_PTS):
        for u in range(N_PTS):
            if p != u:
                Q[p][u] = set()
    
    # Iterate over all q and find pairs (p, u) that form an isosceles triangle with q.
    # Then add q to Q[p][u] and Q[u][p].
    
    for q in range(N_PTS):
        dq = dist2[q, :]
        
        # Case 1: dq[p] == dq[u] (vertex at q)
        # Group points by distance from q
        dist_groups = defaultdict(list)
        for p in range(N_PTS):
            if p != q:
                dist_groups[dq[p]].append(p)
        
        for d, pts in dist_groups.items():
            if len(pts) < 2:
                continue
            # Any pair in pts forms an isosceles triangle with q (vertex at q)
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    p = pts[i]
                    u = pts[j]
                    Q[p][u].add(q)
                    Q[u][p].add(q)
                    
        # Case 2: dq[p] == dist2[p, u] (vertex at p)
        # And Case 3: dq[u] == dist2[p, u] (vertex at u)
        
        # Use NumPy for speed
        # Condition 2: dq[p] == dist2[p, u]
        mask2 = (dq[:, np.newaxis] == dist2)
        mask2[q, :] = False
        mask2[:, q] = False
        np.fill_diagonal(mask2, False)
        
        p_idx, u_idx = np.where(mask2)
        for p, u in zip(p_idx, u_idx):
            Q[p][u].add(q)
            Q[u][p].add(q)
            
        # Condition 3: dq[u] == dist2[p, u]
        mask3 = (dq[np.newaxis, :] == dist2)
        mask3[q, :] = False
        mask3[:, q] = False
        np.fill_diagonal(mask3, False)
        
        p_idx, u_idx = np.where(mask3)
        for p, u in zip(p_idx, u_idx):
            Q[p][u].add(q)
            Q[u][p].add(q)

_start_time = time.time()
_precompute_coverage()

def _greedy_blocking_power(deadline):
    """
    Greedy algorithm that picks the point which reduces the candidate set the most.
    Optimized using precomputed Q sets.
    """
    S = []
    S_set = set()
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
            # Calculate covered points
            covered = set()
            for u in S_set:
                covered.update(Q[p][u])
            
            # New C is C minus covered minus p
            new_C = C - covered - {p}
            
            new_C_size = len(new_C)
            if new_C_size < best_new_C_size:
                best_new_C_size = new_C_size
                best_p = p
                best_new_C = new_C
        
        if best_p == -1:
            break
            
        S.append(best_p)
        S_set.add(best_p)
        C = best_new_C
        
    return S

def _is_maximal(S_set):
    """
    Check if S_set is a maximal isosceles-free set.
    """
    if not S_set:
        return N_PTS == 0
    
    for x in range(N_PTS):
        if x in S_set:
            continue
        
        # Check if x is covered by any pair in S_set
        # x is covered if there exists u, v in S_set such that x in Q[u][v]
        found_conflict = False
        for u in S_set:
            for v in S_set:
                if u < v and x in Q[u][v]:
                    found_conflict = True
                    break
            if found_conflict:
                break
        
        if not found_conflict:
            return False
            
    return True

def _prune_maximal_set(S_list):
    """
    Try to reduce the size of a maximal isosceles-free set.
    """
    S_set = set(S_list)
    changed = True
    while changed:
        changed = False
        for p in list(S_set):
            S_temp = S_set - {p}
            if _is_maximal(S_temp):
                S_set = S_temp
                changed = True
                break
    return list(S_set)

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    best_set = None
    best_size = float('inf')
    
    seed = 42
    while time.time() < deadline - 15.0:
        random.seed(seed)
        candidate = _greedy_blocking_power(deadline - 10.0)
        
        seed += 1
        
        if not candidate:
            continue
        
        pruned_candidate = _prune_maximal_set(candidate)
        
        if len(pruned_candidate) < best_size:
            best_size = len(pruned_candidate)
            best_set = list(pruned_candidate)
            
    if best_set is None:
        # Fallback
        random.seed(0)
        S = []
        S_set = set()
        C = set(range(N_PTS))
        while C:
            p = random.choice(list(C))
            S.append(p)
            S_set.add(p)
            covered = set()
            for u in S_set:
                covered.update(Q[p][u])
            C = C - covered - {p}
        best_set = _prune_maximal_set(S)

    return [ALL_POINTS[i] for i in best_set]
