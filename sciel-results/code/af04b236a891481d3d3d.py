import time
import random
import numpy as np
from itertools import combinations

N = 16
K = 4
ALL_SETS = []
MASKS = []
INDEX_BY_MASK = {}

# Precompute masks
for comb in combinations(range(1, N+1), K):
    mask = 0
    for x in comb:
        mask |= (1 << (x-1))
    MASKS.append(mask)
    ALL_SETS.append(comb)
    INDEX_BY_MASK[mask] = len(MASKS) - 1

M = len(MASKS)
MASKS_ARR = np.array(MASKS, dtype=np.uint32)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_family = []
    best_size = 0
    
    while time.time() < deadline:
        order = np.random.permutation(M)
        current_family_masks = []
        current_family_indices = []
        
        # We will iterate through candidates
        for idx in order:
            if time.time() > deadline - 5.0:
                break
                
            m = MASKS[idx]
            
            # Check if m conflicts with any pair in current_family
            # Conflict: exists i, j in current_family such that
            # (m & i) == (m & j) == (i & j)
            
            if len(current_family_masks) < 2:
                current_family_masks.append(m)
                current_family_indices.append(idx)
                continue
            
            # Convert to numpy array for fast ops
            # This conversion might be slow if done every time.
            # Better to maintain a numpy array and append?
            # Or just loop in Python if N is small (250).
            # 250^2 = 62500 iterations.
            # Python loop of 62500 with bit ops is ~10ms?
            # 1820 candidates * 10ms = 18s per restart. Too slow for many restarts.
            
            # Let's try a faster check.
            # For a fixed m, compute s_i = m & A_i for all i.
            # We need to find i, j such that s_i == s_j == A_i & A_j.
            
            # Group indices by s_i?
            # If s_i == s_j, then we must have A_i & A_j == s_i.
            # Since s_i = m & A_i and s_j = m & A_j, and s_i == s_j,
            # let S_int = s_i.
            # We need A_i & A_j == S_int.
            
            # This grouping might help.
            groups = {}
            for i, A_i in enumerate(current_family_masks):
                s_i = m & A_i
                if s_i not in groups:
                    groups[s_i] = []
                groups[s_i].append(i)
                
            conflict_found = False
            for s_val, indices in groups.items():
                if len(indices) < 2:
                    continue
                # Check pairs within this group
                for p in range(len(indices)):
                    for q in range(p+1, len(indices)):
                        i = indices[p]
                        j = indices[q]
                        # We know m & A_i == s_val and m & A_j == s_val
                        # Check if A_i & A_j == s_val
                        if (current_family_masks[i] & current_family_masks[j]) == s_val:
                            conflict_found = True
                            break
                    if conflict_found:
                        break
                if conflict_found:
                    break
            
            if not conflict_found:
                current_family_masks.append(m)
                current_family_indices.append(idx)
                
        if len(current_family_indices) > best_size:
            best_size = len(current_family_indices)
            best_family = current_family_indices[:]
            
        # Early stop if we hit theoretical max or good enough
        if best_size >= 250:
            break

    result = []
    for idx in best_family:
        result.append(ALL_SETS[idx])
        
    return result
