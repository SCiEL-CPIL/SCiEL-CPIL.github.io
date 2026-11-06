"""Improved program: MODEL-AND-OPTIMIZE for small maximal Sidon sets in [1,256].

Strategy:
1. Start from strong baselines (greedy ascending, descending, random).
2. Use combined local search:
   a. Remove elements one by one if the set remains maximal.
   b. Try swaps: remove one element, add a missing element, check Sidon + maximality.
3. Multiple randomized restarts to escape local optima.
4. Deadline guard with safety margin.
"""

import time
import random

N = 256

# EVOLVE-BLOCK-START

def _is_sidon(set_list):
    """Check if set_list is a Sidon set. Returns True if all pair sums a+b (a<=b) are distinct."""
    s = set(set_list)
    if len(s) != len(set_list):
        return False
    sums = set()
    lst = sorted(set_list)
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            sm = lst[i] + lst[j]
            if sm in sums:
                return False
            sums.add(sm)
    return True

def _compute_sums(set_list):
    """Compute the set of all pair sums."""
    sums = set()
    lst = sorted(set_list)
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            sums.add(lst[i] + lst[j])
    return sums

def _is_maximal_fast(set_list, sums, N_val=N):
    """Faster maximality check using precomputed sums."""
    s = set(set_list)
    for x in range(1, N_val + 1):
        if x in s:
            continue
        if 2 * x in sums:
            continue
        conflict = False
        for elem in set_list:
            if (x + elem) in sums:
                conflict = True
                break
        if not conflict:
            return False
    return True

def _greedy_ascending():
    """Original greedy: pick smallest possible element."""
    selected = []
    sums = set()
    for x in range(1, N + 1):
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _greedy_descending():
    """Greedy descending: pick largest possible element."""
    selected = []
    sums = set()
    for x in range(N, 0, -1):
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _greedy_random(seed):
    """Greedy with random ordering."""
    rng = random.Random(seed)
    order = list(range(1, N + 1))
    rng.shuffle(order)
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _make_maximal(current, deadline):
    """
    Ensure the set is maximal by adding any missing element that doesn't break Sidon.
    Returns the maximal set.
    """
    current = list(current)
    current_set = set(current)
    current_sums = _compute_sums(current)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if x in current_set:
                continue
            
            # Check if adding x keeps it Sidon
            new_sums = [x + a for a in current] + [2 * x]
            # Check for internal conflicts in new_sums
            if len(set(new_sums)) != len(new_sums):
                continue
            # Check conflict with existing sums
            if current_sums.intersection(set(new_sums)):
                continue
            
            # It is a valid addition. Add it.
            current.append(x)
            current_set.add(x)
            current_sums.update(new_sums)
            changed = True
            break
            
    return current

def _local_search_reduce(start_set, deadline, rng):
    """
    Local search: iteratively removes elements that do not break maximality.
    This is the primary optimization step.
    """
    current = list(start_set)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(current)))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            candidate = current[:idx] + current[idx+1:]
            if not candidate:
                continue
            
            cand_sums = _compute_sums(candidate)
            if _is_maximal_fast(candidate, cand_sums):
                current = candidate
                improved = True
                break

    return current

def _local_search_swap(start_set, deadline, rng, best_size):
    """
    Try to improve by swapping: remove one, add another, check Sidon + maximal.
    Accept only if size decreases or if it enables further reduction.
    """
    current = list(start_set)
    current_size = len(current)
    
    if current_size < 2:
        return current
    
    improved = True
    iterations = 0
    while improved and time.time() < deadline and iterations < 20:
        improved = False
        iterations += 1
        current_set = set(current)
        # Try all pairs (remove r, add a)
        remove_candidates = list(current)
        rng.shuffle(remove_candidates)
        add_candidates = [x for x in range(1, N+1) if x not in current_set]
        rng.shuffle(add_candidates)
        
        for r in remove_candidates:
            if time.time() > deadline:
                break
            for a in add_candidates:
                if time.time() > deadline:
                    break
                candidate = [x for x in current if x != r] + [a]
                candidate = sorted(set(candidate))
                if len(candidate) != current_size:
                    continue
                # Check Sidon
                if not _is_sidon(candidate):
                    continue
                # Check maximal
                cand_sums = _compute_sums(candidate)
                if _is_maximal_fast(candidate, cand_sums):
                    # Accept swap (same size but maybe enables further reductions)
                    current = candidate
                    improved = True
                    break
            if improved:
                break
    return current

def construct():
    start_time = time.time()
    # Use 28s deadline to be safe against 30s hard limit
    deadline = start_time + 28.0
    
    # Generate diverse starting sets
    candidates = []
    
    # 1. Greedy ascending
    candidates.append(_greedy_ascending())
    
    # 2. Greedy descending
    candidates.append(_greedy_descending())
    
    # 3. Multiple random greedy (increase count for better coverage, but check time)
    # Greedy is O(N^2) roughly, which is fast. 100 greeds should be < 1s.
    for seed in range(100):
        if time.time() > deadline:
            break
        candidates.append(_greedy_random(seed))
    
    # Filter valid Sidon sets (all greedy outputs are Sidon by construction, but check just in case)
    valid_candidates = []
    for c in candidates:
        if _is_sidon(c):
            valid_candidates.append(c)
            
    if not valid_candidates:
        return _greedy_ascending()
        
    # Sort by size to prioritize smaller sets
    valid_candidates.sort(key=len)
    
    best_set = valid_candidates[0]
    best_size = len(best_set)
    
    # Apply local search to top candidates
    # We process more candidates but with a strict deadline
    rng = random.Random(42)
    
    # Process top 30 candidates (increased from 15, but local search is faster now)
    for c in valid_candidates[:30]:
        if time.time() > deadline:
            break
        
        # Ensure the candidate is maximal before reducing?
        # Greedy sets are maximal by construction?
        # Greedy ascending: picks x if no conflict. At the end, no x can be added. So yes, maximal.
        # Greedy descending: same.
        # Greedy random: same.
        # So we can start reduction directly.
        
        # First, try reduction
        reduced = _local_search_reduce(c, deadline, rng)
        
        # Verify the result is still a valid maximal Sidon set
        if _is_sidon(reduced) and _is_maximal_fast(reduced, _compute_sums(reduced)):
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_set = reduced
        
        # If we still have time, try swaps on the best found so far
        if time.time() < deadline - 5.0 and len(best_set) > 10:
            swapped = _local_search_swap(best_set, deadline, rng, best_size)
            if _is_sidon(swapped) and _is_maximal_fast(swapped, _compute_sums(swapped)):
                if len(swapped) < best_size:
                    best_size = len(swapped)
                    best_set = swapped
    
    # Final verification
    if not _is_sidon(best_set):
        return _greedy_ascending()
        
    # Ensure maximality (safety check)
    sums = _compute_sums(best_set)
    if not _is_maximal_fast(best_set, sums):
        # This should not happen if local search logic is correct, but fix if needed
        best_set = _make_maximal(best_set, deadline)
        
    return best_set

# EVOLVE-BLOCK-END
