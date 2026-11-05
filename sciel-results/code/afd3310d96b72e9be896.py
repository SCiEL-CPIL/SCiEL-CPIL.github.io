"""
Minimum maximal no-isosceles set on [0,N)^2.

Strategy:
1. Precompute isosceles conflict structure.
2. Use a greedy "blocking power" heuristic that picks the point which reduces
   the candidate set the most (mimicking a set-cover/dominating-set approach).
3. Polish the result with local search: try to find smaller maximal sets by
   removing points and re-adding different ones, or by swapping points.
4. Multiple restarts with randomization and perturbation for diversity.
"""

import random
import time
import itertools
import math
import numpy as np
from collections import defaultdict

N = 16
ALL_POINTS = list(itertools.product(range(N), repeat=2))
N_PTS = len(ALL_POINTS)

# Precompute squared distances
coords = np.array(ALL_POINTS, dtype=np.int32)
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist2 = np.sum(diff.astype(np.int64) ** 2, axis=2)

# Precompute bad pairs for each point
# BAD_PAIRS[p] is a set of (i, j) with i < j such that {p, i, j} is isosceles.
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
            BAD_PAIRS[p].add((i_idx[k], j_idx[k]))

_start_time = time.time()
_precompute_bad_pairs()
_precompute_time = time.time() - _start_time

# For fast lookup: for each point q, store BAD_PAIRS[q] as a set of frozenset pairs
# Actually, let's also precompute for each pair (i,j), which points p make {p,i,j} isosceles
# This helps in the local search: if we remove p, which points become addable?

# For each point p, precompute the set of points that become "unblocked" if p is removed
# This is complex; let's use a different approach for local search.

def _is_valid_addition(p, S_set):
    """Check if p can be added to S_set (a set of indices)."""
    for u, v in BAD_PAIRS[p]:
        if u in S_set and v in S_set:
            return False
    return True

def _greedy_blocking_power(deadline, perturb_strength=0.0):
    """
    Greedy algorithm that picks the point which reduces the candidate set the most.
    Uses vectorized numpy operations for speed.
    
    perturb_strength: if > 0, add random noise to blocking power for diversity.
    """
    S = []
    S_set = set()
    C = set(range(N_PTS))
    
    while C:
        if time.time() > deadline:
            break
        
        if not C:
            break
            
        candidates = list(C)
        
        # Compute blocking power for each candidate
        # For each p in C, new_C = {q in C, q != p | q is valid with S U {p}}
        # q is valid with S U {p} iff:
        #   1. q is valid with S (already true since q in C)
        #   2. For all u in S, {q, p, u} is not isosceles
        #      i.e., (p, u) not in BAD_PAIRS[q] for all u in S
        
        # Precompute for each u in S, the set of points q such that (p, u) in BAD_PAIRS[q]
        # Actually, for a fixed p, the points q that are blocked by p (given S) are those
        # where exists u in S such that (min(p,u), max(p,u)) in BAD_PAIRS[q].
        
        # Let's compute for each candidate p, the number of q in C that remain valid.
        # blocking_power[p] = |{q in C, q != p | q valid with S U {p}}|
        
        # To speed up, precompute for each pair (a,b), the set of points q such that (a,b) in BAD_PAIRS[q]
        # pair_to_q[(a,b)] = set of q where (a,b) is a bad pair for q
        
        # This is a large precomputation but done once.
        
        # For now, let's use a simpler approach: for each p in C, iterate over S and check.
        # But we can vectorize the inner check.
        
        # Actually, let's precompute pair_to_q once globally.
        
        best_p = -1
        best_new_C_size = float('inf')
        best_new_C = None
        
        # Shuffle for tie-breaking
        random.shuffle(candidates)
        
        for p in candidates:
            # Compute which q in C are valid with S U {p}
            # q is valid iff for all u in S, (min(p,u), max(p,u)) not in BAD_PAIRS[q]
            
            # Precompute the set of pairs to check: {(min(p,u), max(p,u)) for u in S}
            pairs_to_check = set()
            for u in S:
                a, b = (p, u) if p < u else (u, p)
                pairs_to_check.add((a, b))
            
            # Now for each q in C (q != p), check if any pair in pairs_to_check is in BAD_PAIRS[q]
            new_C = set()
            for q in C:
                if q == p:
                    continue
                # Check if q is blocked by p (given S)
                blocked = False
                for pair in pairs_to_check:
                    if pair in BAD_PAIRS[q]:
                        blocked = True
                        break
                if not blocked:
                    new_C.add(q)
            
            new_C_size = len(new_C)
            
            if perturb_strength > 0:
                # Add noise for diversity
                noisy_size = new_C_size + random.gauss(0, perturb_strength)
            else:
                noisy_size = new_C_size
            
            if noisy_size < best_new_C_size:
                best_new_C_size = noisy_size
                best_p = p
                best_new_C = new_C
        
        if best_p == -1:
            break
            
        S.append(best_p)
        S_set.add(best_p)
        C = best_new_C
        
    return S

def _is_maximal(S):
    """Check if S is maximal (no point can be added)."""
    S_set = set(S)
    for p in range(N_PTS):
        if p in S_set:
            continue
        if _is_valid_addition(p, S_set):
            return False
    return True

def _local_search_polish(S, deadline, max_iterations=50):
    """
    Try to find a smaller maximal set by local search.
    Moves:
    1. Remove a point p from S, then greedily add points to make it maximal again.
       If the result is smaller, accept.
    2. Swap: remove p, add q (not in S), check if still valid and maximal.
    """
    S = list(S)
    S_set = set(S)
    best_size = len(S)
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try removal moves
        for i in range(len(S)):
            p = S[i]
            S_trial = S[:i] + S[i+1:]
            S_trial_set = set(S_trial)
            
            # Check if S_trial is valid (it should be, as a subset of valid S)
            # Now greedily add points to make it maximal
            C = set(range(N_PTS)) - S_trial_set
            new_S = list(S_trial)
            new_S_set = set(S_trial)
            
            while C:
                if time.time() > deadline:
                    break
                
                # Find a point that can be added
                # Use blocking power heuristic
                best_q = -1
                best_new_C_size = float('inf')
                
                candidates = list(C)
                random.shuffle(candidates)
                
                for q in candidates:
                    if not _is_valid_addition(q, new_S_set):
                        continue
                    
                    # Compute new C if we add q
                    new_C = set()
                    for r in C:
                        if r == q:
                            continue
                        # Check if r is valid with new_S U {q}
                        valid = True
                        for u in new_S:
                            a, b = (q, u) if q < u else (u, q)
                            if (a, b) in BAD_PAIRS[r]:
                                valid = False
                                break
                        if valid:
                            new_C.add(r)
                    
                    new_C_size = len(new_C)
                    if new_C_size < best_new_C_size:
                        best_new_C_size = new_C_size
                        best_q = q
                
                if best_q == -1:
                    break
                    
                new_S.append(best_q)
                new_S_set.add(best_q)
                C = new_C
            
            # Check if new_S is maximal
            if _is_maximal(new_S) and len(new_S) < best_size:
                S = new_S
                S_set = set(S)
                best_size = len(S)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swap moves
        for i in range(len(S)):
            p = S[i]
            S_without_p = set(S) - {p}
            
            # Find points q not in S that can replace p
            for q in range(N_PTS):
                if q in S_set:
                    continue
                
                # Check if S - {p} + {q} is valid
                S_trial_set = S_without_p | {q}
                
                # Check validity: for all triples in S_trial_set, not isosceles
                # This is O(k^3) but k is small
                valid = True
                S_list = list(S_trial_set)
                for idx1 in range(len(S_list)):
                    for idx2 in range(idx1 + 1, len(S_list)):
                        for idx3 in range(idx2 + 1, len(S_list)):
                            a, b, c = S_list[idx1], S_list[idx2], S_list[idx3]
                            # Check if {a,b,c} is isosceles
                            d_ab = dist2[a, b]
                            d_ac = dist2[a, c]
                            d_bc = dist2[b, c]
                            if d_ab == d_ac or d_ab == d_bc or d_ac == d_bc:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Check if S_trial is maximal
                if _is_maximal(list(S_trial_set)) and len(S_trial_set) < best_size:
                    S = list(S_trial_set)
                    S_set = set(S)
                    best_size = len(S)
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    best_set = None
    best_size = float('inf')
    
    # Multiple restarts with varying perturbation
    restart_count = 0
    while time.time() < deadline - 15.0:
        restart_count += 1
        
        # Vary perturbation strength for diversity
        perturb = 0.0 if restart_count % 3 == 0 else random.uniform(0.5, 2.0)
        
        candidate = _greedy_blocking_power(deadline - 10.0, perturb_strength=perturb)
        
        if candidate is None or len(candidate) == 0:
            continue
        
        # Verify validity
        c_set = set(candidate)
        valid = True
        for p in candidate:
            if not _is_valid_addition(p, c_set - {p}):
                valid = False
                break
        
        if not valid:
            continue
        
        # Check maximality
        if not _is_maximal(candidate):
            # Try to make it maximal by adding points
            C = set(range(N_PTS)) - c_set
            while C:
                if time.time() > deadline - 5.0:
                    break
                added = False
                for q in list(C):
                    if _is_valid_addition(q, c_set):
                        candidate.append(q)
                        c_set.add(q)
                        C.remove(q)
                        added = True
                        break
                if not added:
                    break
        
        # Local search polish
        polished = _local_search_polish(candidate, deadline - 5.0, max_iterations=20)
        
        if len(polished) < best_size:
            best_size = len(polished)
            best_set = polished
    
    if best_set is None:
        # Fallback: simple greedy
        S = []
        S_set = set()
        C = set(range(N_PTS))
        while C:
            p = random.choice(list(C))
            S.append(p)
            S_set.add(p)
            new_C = set()
            for q in C:
                if q == p:
                    continue
                valid = True
                for u in S:
                    if u == p:
                        continue
                    a, b = (p, u) if p < u else (u, p)
                    if (a, b) in BAD_PAIRS[q]:
                        valid = False
                        break
                if valid:
                    new_C.add(q)
            C = new_C
        best_set = S

    return [ALL_POINTS[i] for i in best_set]
