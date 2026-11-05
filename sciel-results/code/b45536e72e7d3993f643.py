"""Minimum maximal no-isosceles set on [0,N)^2 using Simulated Annealing."""
import random
import time
import itertools
import math

N = 16

# Precompute all grid points
ALL_POINTS = list(itertools.product(range(N), repeat=2))
N_PTS = len(ALL_POINTS)
POINT_TO_IDX = {p: i for i, p in enumerate(ALL_POINTS)}

def _is_iso_fast(a, b, c):
    ab2 = (a[0]-b[0])**2 + (a[1]-b[1])**2
    ac2 = (a[0]-c[0])**2 + (a[1]-c[1])**2
    bc2 = (b[0]-c[0])**2 + (b[1]-c[1])**2
    return ab2 == ac2 or ab2 == bc2 or ac2 == bc2

def _compute_energy(chosen):
    """
    Compute energy for a given set.
    Energy = size + penalty_invalid + penalty_not_maximal
    We want to minimize this.
    """
    chosen_set = set(chosen)
    energy = len(chosen)
    
    # Check validity: no 3 points form isosceles
    # This is O(k^3) which is slow. We need a faster check or incremental update.
    # For SA, we need fast evaluation.
    # Let's precompute conflicts.
    # Conflict is a triple.
    # Actually, checking validity from scratch is O(k^3). If k~50, 125k ops. 256 points.
    # If we do 100k steps, that's 1.25e10 ops. Too slow in Python.
    
    # We need a smarter energy evaluation or a different approach.
    # Let's use a "repair" strategy or keep track of conflicts incrementally.
    
    # Alternative: 
    # 1. Start with a valid maximal set from Greedy.
    # 2. Use Local Search that only allows moves that maintain validity (or repair immediately).
    
    # Let's go back to Local Search but with a different neighborhood: 
    # "2-opt" style: Remove 2 points, add 2 points. Or Remove 1, Add 2.
    # Or use "Tabu Search" on the Greedy+Local Search base.
    
    # The prompt asks for a MATERIALLY DIFFERENT strategy.
    # Greedy + Local Search (Remove/Swap) is the current one.
    # Simulated Annealing with *incremental* validity check is a good candidate.
    
    # To make it fast, we can precompute for each point p, the set of pairs (i,j) 
    # that form an isosceles triangle with p.
    # If we maintain the set S, we can check if adding p is valid by checking
    # if any pair in S is in conflict[p].
    
    # Precomputation:
    # conflict[p] = set of (i, j) such that p, ALL_POINTS[i], ALL_POINTS[j] form isosceles.
    # This is 256 * C(256,2) ~ 8 million checks. Feasible at startup.
    
    pass

# Precompute conflict structure
# conflict_idx[p_idx] = set of (i_idx, j_idx) where i<j and p, i, j form isosceles
# Actually, we need to know if adding p creates a conflict with existing pairs in S.
# So for each p, we store the set of pairs (i,j) that are "bad" with p.
# If S contains both i and j, and (i,j) is in bad_pairs[p], then p cannot be added.

BAD_PAIRS = [set() for _ in range(N_PTS)]

def _precompute_bad_pairs():
    for p_idx, p in enumerate(ALL_POINTS):
        for i in range(N_PTS):
            for j in range(i+1, N_PTS):
                if _is_iso_fast(p, ALL_POINTS[i], ALL_POINTS[j]):
                    BAD_PAIRS[p_idx].add((i, j))

# This precomputation is O(N^3) = 256^3 ~ 16 million. 
# In Python, this might take a few seconds.
# Let's optimize: 
# For each p, iterate i, j.
# We can use numpy to vectorize distance checks?
# Or just do it in pure python but efficiently.

def _precompute_bad_pairs_fast():
    # Use numpy for speed
    import numpy as np
    coords = np.array(ALL_POINTS)
    # dist2[i, j] = squared distance
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    
    for p_idx in range(N_PTS):
        # p is coords[p_idx]
        # We need pairs (i, j) such that dist(p,i) == dist(p,j) or dist(p,i) == dist(i,j) or dist(p,j) == dist(i,j)
        # Let dp_i = dist2[p_idx, i]
        # Let dp_j = dist2[p_idx, j]
        # Let dij = dist2[i, j]
        # Condition: dp_i == dp_j OR dp_i == dij OR dp_j == dij
        
        dp = dist2[p_idx, :]
        # We need to find i < j satisfying the condition.
        # This is still O(N^2) per p.
        # Total O(N^3).
        # 16M iterations in numpy loops? No, we can vectorize the check for a fixed p.
        
        # For fixed p, dp is a vector of size N.
        # dist2 is N x N.
        # We want mask[i, j] where i < j and (dp[i]==dp[j] or dp[i]==dist2[i,j] or dp[j]==dist2[i,j])
        
        # Create upper triangle indices
        i_idx, j_idx = np.triu_indices(N_PTS, k=1)
        
        # dp_i = dp[i_idx], dp_j = dp[j_idx]
        # dij = dist2[i_idx, j_idx]
        
        # cond = (dp[i_idx] == dp[j_idx]) | (dp[i_idx] == dist2[i_idx, j_idx]) | (dp[j_idx] == dist2[i_idx, j_idx])
        
        # This creates large arrays. N_PTS=256. i_idx size ~ 32k.
        # 256 * 32k = 8M operations. This is fine.
        
        dp_i = dp[i_idx]
        dp_j = dp[j_idx]
        dij = dist2[i_idx, j_idx]
        
        cond = (dp_i == dp_j) | (dp_i == dij) | (dp_j == dij)
        
        # Get indices where cond is True
        bad_indices = np.where(cond)[0]
        # Convert back to (i, j) pairs
        for k in bad_indices:
            BAD_PAIRS[p_idx].add((i_idx[k], j_idx[k]))

# Run precomputation
_start_time = time.time()
_precompute_bad_pairs_fast()
_precompute_time = time.time() - _start_time

def _can_add_fast(p_idx, chosen_indices):
    """Check if p_idx can be added to chosen_indices (list of indices)."""
    chosen_set = set(chosen_indices)
    bad_pairs = BAD_PAIRS[p_idx]
    for i, j in bad_pairs:
        if i in chosen_set and j in chosen_set:
            return False
    return True

def _is_maximal_fast(chosen_indices):
    """Check if set is maximal."""
    chosen_set = set(chosen_indices)
    for p_idx in range(N_PTS):
        if p_idx in chosen_set:
            continue
        if _can_add_fast(p_idx, chosen_indices):
            return False
    return True

def _greedy_random():
    order = list(range(N_PTS))
    random.shuffle(order)
    chosen = []
    for p_idx in order:
        if _can_add_fast(p_idx, chosen):
            chosen.append(p_idx)
    return chosen

def _local_search(current, deadline, max_iters=100):
    """Local search: try to remove points."""
    best = list(current)
    best_size = len(best)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        
        # Try removing each point
        for i in range(len(best)):
            if time.time() > deadline:
                return best
            test = best[:i] + best[i+1:]
            if _is_maximal_fast(test):
                best = test
                best_size = len(best)
                improved = True
                break
        
        if improved:
            continue
        
        # If can't remove, try swaps
        for i in range(len(best)):
            if time.time() > deadline:
                return best
            base = best[:i] + best[i+1:]
            base_set = set(base)
            for q in range(N_PTS):
                if q in base_set:
                    continue
                if _can_add_fast(q, base):
                    test = base + [q]
                    if _is_maximal_fast(test):
                        best = test
                        best_size = len(best)
                        improved = True
                        break
            if improved:
                break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    best = None
    best_size = float('inf')
    
    n_restarts = 0
    while time.time() < deadline - 30.0:
        candidate = _greedy_random()
        improved = _local_search(candidate, deadline - 10.0, max_iters=20)
        if len(improved) < best_size:
            best_size = len(improved)
            best = improved
        n_restarts += 1
    
    if best is None:
        best = _greedy_random()
        
    # Convert indices to points
    return [ALL_POINTS[i] for i in best]
