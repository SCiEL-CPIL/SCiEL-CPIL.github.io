"""Improved Local Search with 2-1 Swaps for ternary covering codes K_3(7,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def _precompute_neighbors(q, n, radius):
    """Precompute neighbors for all words."""
    N = q ** n
    nbrs_list = [None] * N
    
    if q == 3 and n == 7:
        for idx in range(N):
            word = [0] * 7
            tmp = idx
            word[0] = tmp % 3; tmp //= 3
            word[1] = tmp % 3; tmp //= 3
            word[2] = tmp % 3; tmp //= 3
            word[3] = tmp % 3; tmp //= 3
            word[4] = tmp % 3; tmp //= 3
            word[5] = tmp % 3; tmp //= 3
            word[6] = tmp % 3; tmp //= 3
            
            nbrs = [idx]
            for pos in range(7):
                old = word[pos]
                for v in range(3):
                    if v != old:
                        new_word = word.copy()
                        new_word[pos] = v
                        new_idx = new_word[0] + new_word[1]*3 + new_word[2]*9 + new_word[3]*27 + new_word[4]*81 + new_word[5]*243 + new_word[6]*729
                        nbrs.append(new_idx)
            nbrs_list[idx] = np.array(nbrs, dtype=np.int32)
    else:
        for idx in range(N):
            word = [0] * n
            tmp = idx
            for i in range(n):
                word[i] = tmp % q
                tmp //= q
            nbrs = [idx]
            for pos in range(n):
                old = word[pos]
                for v in range(q):
                    if v != old:
                        new_word = word.copy()
                        new_word[pos] = v
                        new_idx = 0
                        mul = 1
                        for i in range(n):
                            new_idx += new_word[i] * mul
                            mul *= q
                        nbrs.append(new_idx)
            nbrs_list[idx] = np.array(nbrs, dtype=np.int32)
            
    return nbrs_list

def _greedy_construct(nbrs_list, N, rng):
    """Construct an initial feasible code using randomized greedy."""
    covered_count = np.zeros(N, dtype=np.int32)
    selected = []
    selected_set = set()
    
    # Precompute neighbor matrix for vectorized gain calculation
    max_nbrs = len(nbrs_list[0])
    nbrs_mat = np.zeros((N, max_nbrs), dtype=np.int32)
    for w in range(N):
        nbrs_mat[w, :len(nbrs_list[w])] = nbrs_list[w]
        
    while True:
        is_uncovered = (covered_count == 0).astype(np.int32)
        gain = is_uncovered[nbrs_mat].sum(axis=1)
        
        max_gain = np.max(gain)
        if max_gain == 0:
            break
        
        candidates = np.where(gain == max_gain)[0]
        idx = rng.choice(candidates)
        
        selected.append(idx)
        selected_set.add(idx)
        
        nbrs_idx = nbrs_list[idx]
        covered_count[nbrs_idx] += 1
        
    return selected, covered_count

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    
    nbrs_list = _precompute_neighbors(q, n, radius)
    N = q ** n
    
    # Precompute reverse neighbors: rev_nbrs[v] = list of c such that v in N(c)
    rev_nbrs = [[] for _ in range(N)]
    for c in range(N):
        for v in nbrs_list[c]:
            rev_nbrs[v].append(c)
    # Convert to lists of ints, already done.
    
    best_code = None
    best_size = float('inf')
    
    # Run multiple iterations with different random seeds for the greedy phase
    for seed in range(500): # Max 500 restarts
        if time.time() - start_time > time_limit - 10.0:
            break
            
        rng = np.random.RandomState(seed)
        
        # 1. Greedy Phase
        selected, covered_count = _greedy_construct(nbrs_list, N, rng)
        
        # 2. Local Search Phase
        improved = True
        while improved and time.time() - start_time < time_limit - 5.0:
            improved = False
            
            # 1. Remove redundant
            for i in range(len(selected)):
                w = selected[i]
                nbrs_w = nbrs_list[w]
                if not np.any(covered_count[nbrs_w] == 1):
                    covered_count[nbrs_w] -= 1
                    selected.pop(i)
                    improved = True
                    break
            
            if time.time() - start_time > time_limit - 5.0:
                break
                
            # 2. Try 2-1 Swaps
            # We will try a random sample of pairs to keep it fast
            # Or all pairs if size is small.
            if len(selected) < 300:
                pairs = list(combinations(range(len(selected)), 2))
                rng.shuffle(pairs)
            else:
                # Sample 1000 random pairs
                pairs = []
                for _ in range(1000):
                    i, j = rng.choice(len(selected), 2, replace=False)
                    if i > j: i, j = j, i
                    pairs.append((i, j))
            
            swap_found = False
            for i, j in pairs:
                if time.time() - start_time > time_limit - 5.0:
                    break
                
                c1 = selected[i]
                c2 = selected[j]
                
                n1 = nbrs_list[c1]
                n2 = nbrs_list[c2]
                
                # Create mask for points in N(c1) U N(c2)
                mask = np.zeros(N, dtype=bool)
                mask[n1] = True
                mask[n2] = True
                
                # Calculate remaining coverage for these points
                rem = covered_count.copy()
                rem[n1] -= 1
                rem[n2] -= 1
                
                U_mask = mask & (rem == 0)
                U_indices = np.where(U_mask)[0]
                
                if len(U_indices) == 0:
                    # Can remove both
                    covered_count[n1] -= 1
                    covered_count[n2] -= 1
                    if i > j:
                        i, j = j, i
                    selected.pop(i)
                    selected.pop(j)
                    improved = True
                    swap_found = True
                    break
                else:
                    # Need to find c3 that covers all U_indices
                    # c3 must be in intersection of rev_nbrs[v] for v in U_indices
                    # And c3 must not be in selected.
                    
                    # Find intersection
                    if len(U_indices) > 0:
                        candidates = set(rev_nbrs[U_indices[0]])
                        for v in U_indices[1:]:
                            candidates &= set(rev_nbrs[v])
                            if not candidates:
                                break
                        
                        if candidates:
                            # Filter out selected
                            candidates -= set(selected)
                            if candidates:
                                # Pick one (random or first)
                                c3 = list(candidates)[0]
                                # Perform swap
                                covered_count[n1] -= 1
                                covered_count[n2] -= 1
                                covered_count[nbrs_list[c3]] += 1
                                
                                if i > j:
                                    i, j = j, i
                                selected.pop(i)
                                selected.pop(j)
                                selected.append(c3)
                                
                                improved = True
                                swap_found = True
                                break
            
            if swap_found:
                continue # Restart loop to check for new removals/swaps
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_code = selected.copy()
            
    # Convert indices to words
    res = []
    for idx in best_code:
        word = [0] * n
        tmp = idx
        for i in range(n):
            word[i] = tmp % q
            tmp //= q
        res.append(word)
        
    return res
# EVOLVE-BLOCK-END
