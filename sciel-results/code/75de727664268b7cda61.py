"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
import numpy as np

def _greedy_build(candidates, counts_init=None):
    """Greedy: add candidates in given order if B2[2] constraint holds."""
    chosen = []
    counts = defaultdict(int) if counts_init is None else dict(counts_init)
    for x in candidates:
        # Check all new sums
        ok = True
        for y in chosen:
            s = x + y
            if counts[s] + 1 > 2:
                ok = False
                break
        if ok:
            s2 = 2 * x
            if counts[s2] + 1 > 2:
                ok = False
        if ok:
            for y in chosen:
                counts[x + y] += 1
            counts[2 * x] += 1
            chosen.append(x)
    return chosen, counts

def _is_valid_b2(chosen):
    """Verify B2[2] property."""
    if not chosen:
        return True
    arr = np.array(chosen)
    i_idx, j_idx = np.triu_indices(len(arr))
    sums = arr[i_idx] + arr[j_idx]
    unique, counts_arr = np.unique(sums, return_counts=True)
    return np.all(counts_arr <= 2)

def _fast_check_insert(arr, y, counts_arr):
    """Check if y can be added to set arr given sum counts in counts_arr (numpy array indexed by sum value).
    Returns True if valid."""
    # New sums: y + a for all a in arr, and 2*y
    new_sums = arr + y
    # Check each new sum
    for s in new_sums:
        if counts_arr[s] + 1 > 2:
            return False
    if counts_arr[2 * y] + 1 > 2:
        return False
    return True

def _local_search_insert(chosen, time_limit=10.0):
    """Try to insert new elements to grow the set. Optimized with numpy."""
    chosen = sorted(chosen)
    n = len(chosen)
    start_time = time.time()
    
    # Build counts array indexed by sum value (max sum = 2000)
    max_sum = 2000
    counts_arr = np.zeros(max_sum + 1, dtype=np.int32)
    arr = np.array(chosen, dtype=np.int32)
    
    # Compute initial counts
    i_idx, j_idx = np.triu_indices(n)
    sums = arr[i_idx] + arr[j_idx]
    np.add.at(counts_arr, sums, 1)
    
    chosen_set = set(chosen)
    all_candidates = list(range(1, 1001))
    rng_ls = random.Random(99999)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        n = len(chosen)
        arr = np.array(chosen, dtype=np.int32)
        
        # Recompute counts if size changed
        counts_arr = np.zeros(max_sum + 1, dtype=np.int32)
        i_idx, j_idx = np.triu_indices(n)
        sums = arr[i_idx] + arr[j_idx]
        np.add.at(counts_arr, sums, 1)
        
        # Try inserting candidates in random order
        candidates = [c for c in all_candidates if c not in chosen_set]
        rng_ls.shuffle(candidates)
        
        for y in candidates:
            if time.time() - start_time > time_limit:
                break
            
            # Fast check using numpy
            if _fast_check_insert(arr, y, counts_arr):
                # Insert y
                chosen.append(y)
                chosen_set.add(y)
                improved = True
                # Update counts
                new_arr = np.array(chosen, dtype=np.int32)
                new_sums = new_arr + y
                np.add.at(counts_arr, new_sums, 1)
                counts_arr[2 * y] += 1
        
    return sorted(chosen)

def _local_search_swap(chosen, time_limit=10.0):
    """Try swaps to rearrange the set. Optimized version."""
    chosen = sorted(chosen)
    n = len(chosen)
    start_time = time.time()
    
    max_sum = 2000
    counts_arr = np.zeros(max_sum + 1, dtype=np.int32)
    arr = np.array(chosen, dtype=np.int32)
    i_idx, j_idx = np.triu_indices(n)
    sums = arr[i_idx] + arr[j_idx]
    np.add.at(counts_arr, sums, 1)
    
    chosen_set = set(chosen)
    all_candidates = list(range(1, 1001))
    rng_ls = random.Random(55555)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        n = len(chosen)
        arr = np.array(chosen, dtype=np.int32)
        
        # Recompute counts
        counts_arr = np.zeros(max_sum + 1, dtype=np.int32)
        i_idx, j_idx = np.triu_indices(n)
        sums = arr[i_idx] + arr[j_idx]
        np.add.at(counts_arr, sums, 1)
        
        # Try removing each element and replacing with a better one
        candidates = [c for c in all_candidates if c not in chosen_set]
        rng_ls.shuffle(candidates)
        
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            x = chosen[i]
            
            # Remove x from counts
            counts_arr[2 * x] -= 1
            for j in range(n):
                if j != i:
                    counts_arr[x + chosen[j]] -= 1
            
            # Try to find a replacement
            found = False
            for y in candidates:
                if y == x:
                    continue
                # Check if y can be added to the remaining set
                remaining = [chosen[j] for j in range(n) if j != i]
                rem_arr = np.array(remaining, dtype=np.int32)
                
                ok = True
                for s in rem_arr + y:
                    if counts_arr[s] + 1 > 2:
                        ok = False
                        break
                if ok and counts_arr[2 * y] + 1 > 2:
                    ok = False
                
                if ok:
                    # Do the swap
                    chosen[i] = y
                    chosen_set.discard(x)
                    chosen_set.add(y)
                    improved = True
                    found = True
                    break
            
            if not found:
                # Restore x
                counts_arr[2 * x] += 1
                for j in range(n):
                    if j != i:
                        counts_arr[x + chosen[j]] += 1
            
            if found:
                break
    
    return sorted(chosen)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s
    
    # Strategy 1: Standard greedy (increasing order)
    candidates = list(range(1, 1001))
    chosen1, _ = _greedy_build(candidates)
    
    best = chosen1[:]
    best_size = len(best)
    
    # Strategy 2: Try structured and random orderings
    rng = random.Random(42)
    
    # Approach B: Large numbers first
    if time.time() < deadline:
        cand_large = list(range(1000, 0, -1))
        chosen_large, _ = _greedy_build(cand_large)
        if len(chosen_large) > best_size:
            best = chosen_large[:]
            best_size = len(best)
    
    # Approach C: Mixed - alternate small and large
    if time.time() < deadline:
        small = list(range(1, 501))
        large = list(range(500, 0, -1))
        cand_mixed = []
        si, li = 0, 0
        while si < len(small) or li < len(large):
            if si < len(small):
                cand_mixed.append(small[si])
                si += 1
            if li < len(large):
                cand_mixed.append(large[li])
                li += 1
        chosen_mixed, _ = _greedy_build(cand_mixed)
        if len(chosen_mixed) > best_size:
            best = chosen_mixed[:]
            best_size = len(best)
    
    # Strategy 3: Try random orderings with bias
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart % 3 == 0:
            cand = list(range(1, 1001))
            rng.shuffle(cand)
        elif restart % 3 == 1:
            biased = [(rng.random() ** 0.5, i) for i in range(1000)]
            biased.sort(key=lambda x: x[0])
            cand = [b[1] + 1 for b in biased]
        else:
            biased = [(rng.random() ** 0.3, i) for i in range(1000)]
            biased.sort(key=lambda x: x[0])
            cand = [b[1] + 1 for b in biased]
        
        chosen_r, _ = _greedy_build(cand)
        if len(chosen_r) > best_size:
            best = chosen_r[:]
            best_size = len(best)
    
    # Local search: first try to insert new elements (grow size)
    if time.time() < deadline - 5.0:
        ls_time = min(20.0, deadline - time.time() - 3.0)
        if ls_time > 2.0:
            best = _local_search_insert(best, time_limit=ls_time)
            best_size = len(best)
    
    # Then try swaps to potentially enable more insertions
    if time.time() < deadline - 3.0:
        ls_time = min(15.0, deadline - time.time() - 2.0)
        if ls_time > 1.0:
            best = _local_search_swap(best, time_limit=ls_time)
            best_size = len(best)
    
    # Final insertion pass after swaps
    if time.time() < deadline - 2.0:
        ls_time = min(10.0, deadline - time.time() - 1.0)
        if ls_time > 0.5:
            best = _local_search_insert(best, time_limit=ls_time)
            best_size = len(best)
    
    # Verify validity
    if not _is_valid_b2(best):
        best = chosen1[:]
    
    return best
# EVOLVE-BLOCK-END
