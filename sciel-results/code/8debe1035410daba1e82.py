"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256
import time
import random
import sys

# EVOLVE-BLOCK-START
def _is_sidon(selected):
    """Check if a set is Sidon (all pairwise sums distinct, incl. diagonals)."""
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_sums(selected):
    """Get all pairwise sums a+b with a<=b."""
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            sums.add(a + b)
    return sums

def _can_add(sums, selected, x):
    """Can x be added to selected (given its sums) while staying Sidon?"""
    for a in selected:
        s = a + x
        if s in sums:
            return False
    if 2 * x in sums:
        return False
    return True

def _is_maximal_fast(selected, sums):
    """Check maximality: no missing element can be added. Early exit."""
    selected_set = set(selected)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        if _can_add(sums, selected, x):
            return False
    return True

def _greedy_construct(order, N):
    """Greedy construction using a given order; runs to completion -> maximal."""
    selected = []
    sums = set()
    for x in order:
        if _can_add(sums, selected, x):
            selected.append(x)
            for a in selected[:-1]:
                sums.add(a + x)
            sums.add(2 * x)
    return selected

def _complete_to_maximal(selected):
    """
    Takes a Sidon set and greedily adds elements to make it maximal.
    Returns the maximal Sidon set.
    """
    selected = list(selected)
    sums = _get_sums(selected)
    selected_set = set(selected)
    
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        if _can_add(sums, selected, x):
            selected.append(x)
            selected_set.add(x)
            for a in selected[:-1]:
                sums.add(a + x)
            sums.add(2 * x)
            
    return selected

def _local_search_minimize(selected, deadline):
    """
    Local search to minimize a maximal Sidon set.
    Tries to remove elements one by one, and if removal breaks maximality,
    tries to find a replacement or re-add in a different order.
    """
    selected = list(selected)
    best = list(selected)
    best_size = len(selected)
    
    while time.time() < deadline:
        improved = False
        for i in range(len(selected)):
            if time.time() >= deadline:
                break
            # Try removing element i
            trial = selected[:i] + selected[i+1:]
            if not trial:
                continue
            trial_sums = _get_sums(trial)
            # Check if trial is still maximal
            if _is_maximal_fast(trial, trial_sums):
                selected = trial
                best = list(selected)
                best_size = len(selected)
                improved = True
                break
        if not improved:
            break
    
    return best

def _randomized_greedy(rng, N):
    """Randomized greedy: shuffle the order and greedily construct."""
    order = list(range(1, N + 1))
    rng.shuffle(order)
    return _greedy_construct(order, N)

def _backtracking_search(k, deadline, rng):
    """
    Try to find a maximal Sidon set of exactly size k using backtracking.
    Returns the set if found, None otherwise.
    """
    def backtrack(current, current_sums, start_idx):
        if len(current) == k:
            # Check if maximal
            if _is_maximal_fast(current, current_sums):
                return list(current)
            return None
        
        if time.time() >= deadline:
            return None
        
        # Try adding elements from start_idx onwards
        for x in range(start_idx, N + 1):
            if not _can_add(current_sums, current, x):
                continue
            
            new_current = current + [x]
            new_sums = set(current_sums)
            for a in current:
                new_sums.add(a + x)
            new_sums.add(2 * x)
            
            result = backtrack(new_current, new_sums, x + 1)
            if result is not None:
                return result
        
        return None
    
    return backtrack([], set(), 1)

def construct():
    deadline = time.time() + 27  # safety margin before 30s kill
    rng = random.Random(12345)
    
    best_maximal = None
    best_size = float('inf')
    
    # 1. Get initial solutions from multiple randomized greedy runs
    candidates = []
    # Ascending greedy
    candidates.append(_greedy_construct(list(range(1, N + 1)), N))
    # Descending greedy
    candidates.append(_greedy_construct(list(range(N, 0, -1)), N))
    # Several randomized greedy
    for _ in range(5):
        candidates.append(_randomized_greedy(rng, N))
    
    for cand in candidates:
        if len(cand) < best_size:
            best_size = len(cand)
            best_maximal = list(cand)
    
    # 2. Local search on the best candidate
    if best_maximal is not None:
        local_best = _local_search_minimize(best_maximal, deadline)
        if len(local_best) < best_size:
            best_size = len(local_best)
            best_maximal = local_best
    
    # 3. Try backtracking search for small k
    # Start from a reasonable lower bound
    # For Sidon sets in [1,n], max size is about sqrt(n)
    # Maximal Sidon sets can be smaller
    # Try k from max(5, best_size - 10) down to 5
    min_k_to_try = max(5, best_size - 15)
    max_k_to_try = best_size - 1
    
    for k in range(max_k_to_try, min_k_to_try - 1, -1):
        if time.time() >= deadline - 2:
            break
        result = _backtracking_search(k, deadline - 2, rng)
        if result is not None:
            if len(result) < best_size:
                best_size = len(result)
                best_maximal = result
            break  # Found one, stop
    
    # 4. More randomized greedy + local search
    while time.time() < deadline - 1:
        cand = _randomized_greedy(rng, N)
        if len(cand) < best_size:
            best_size = len(cand)
            best_maximal = list(cand)
        
        # Local search on this candidate
        local_best = _local_search_minimize(cand, min(deadline - 1, time.time() + 2))
        if len(local_best) < best_size:
            best_size = len(local_best)
            best_maximal = local_best
    
    # Final validation
    if best_maximal is not None:
        if _is_sidon(best_maximal) and _is_maximal_fast(best_maximal, _get_sums(best_maximal)):
            return sorted(best_maximal)
    
    # Fallback
    return sorted(_greedy_construct(list(range(1, N + 1)), N))
# EVOLVE-BLOCK-END
