import time
import random
import numpy as np
from itertools import combinations

# Precompute all 4-subsets of [1,16] as bitmasks
_ALL_MASKS = []
_MASK_TO_IDX = {}
for c in combinations(range(16), 4):
    mask = 0
    for i in c:
        mask |= (1 << i)
    _ALL_MASKS.append(mask)
    _MASK_TO_IDX[mask] = len(_ALL_MASKS) - 1

_N = len(_ALL_MASKS)  # 1820

# Precompute pairwise intersection masks
# inter[i, j] = _ALL_MASKS[i] & _ALL_MASKS[j]
inter = np.zeros((_N, _N), dtype=np.int32)
for i in range(_N):
    mi = _ALL_MASKS[i]
    for j in range(_N):
        mj = _ALL_MASKS[j]
        inter[i, j] = mi & mj

def _can_add(c, F):
    """
    Check if candidate c can be added to family F (list of indices).
    Returns True if valid.
    """
    if not F:
        return True
    
    F_arr = np.array(F)
    
    # Intersections of each set in F with c
    K = inter[F_arr, c]
    
    # Group indices in F by their intersection value with c
    # We look for i, j in F, i != j, such that:
    # inter[F[i]][F[j]] == K[i] == K[j]
    
    unique_k, inverse = np.unique(K, return_inverse=True)
    
    for idx_k, k in enumerate(unique_k):
        # Find positions in F_arr where K == k
        positions = np.where(K == k)[0]
        
        if len(positions) < 2:
            continue
        
        # For each i in positions, check if there exists j in positions, j != i
        # such that inter[F[i]][F[j]] == k
        
        # Optimization: if k == 0, we are checking for disjoint pairs in J_0
        # If k != 0, J_k is likely small
        
        for p_i in positions:
            i_idx = F_arr[p_i]
            # Check other positions
            for p_j in positions:
                if p_i == p_j:
                    continue
                j_idx = F_arr[p_j]
                if inter[i_idx, j_idx] == k:
                    return False
                    
    return True

def _greedy_build(order, deadline):
    """Build a family using greedy addition in the given order."""
    chosen = []
    for idx in order:
        if time.time() > deadline:
            break
        if _can_add(idx, chosen):
            chosen.append(idx)
    return chosen

def _local_improve(family_indices, deadline):
    """Try to improve the family by swapping out sets to allow more additions."""
    family = list(family_indices)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(family)):
            if time.time() > deadline:
                break
            removed = family[i]
            remaining = family[:i] + family[i+1:]
            # Try to add any set that can be added to remaining
            # We prioritize sets that were previously blocked?
            # Just try random ones or all?
            # Let's try all, but break early if found
            for idx in range(_N):
                if idx == removed:
                    continue
                if idx in remaining:
                    continue
                if _can_add(idx, remaining):
                    family[i] = idx
                    improved = True
                    break
            if improved:
                break # Restart the loop as family changed
    return family

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    best_family = []
    
    # Strategy 1: Lexicographic order
    lex_order = list(range(_N))
    family1 = _greedy_build(lex_order, deadline)
    if len(family1) > len(best_family):
        best_family = family1
        
    # Strategy 2: Multiple random restarts
    rng = random.Random(42)
    num_restarts = 500 # Can be more if fast
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        order = list(range(_N))
        rng.shuffle(order)
        family = _greedy_build(order, deadline)
        if len(family) > len(best_family):
            best_family = family
            
    # Strategy 3: Local improvement
    if time.time() < deadline:
        improved = _local_improve(list(best_family), deadline)
        if len(improved) > len(best_family):
            best_family = improved
            
    # Convert back to sorted lists of 1-based integers
    result = []
    for idx in best_family:
        mask = _ALL_MASKS[idx]
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result
