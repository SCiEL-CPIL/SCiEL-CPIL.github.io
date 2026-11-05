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

def _complete_to_maximal(selected, order=None):
    """
    Takes a Sidon set and greedily adds elements to make it maximal.
    If order is given, use that order; otherwise use 1..N.
    Returns the maximal Sidon set.
    """
    selected = list(selected)
    sums = _get_sums(selected)
    selected_set = set(selected)
    
    if order is None:
        order = list(range(1, N + 1))
    
    for x in order:
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
    Tries to remove elements one by one.
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

def _swap_remove_search(selected, deadline, rng):
    """
    Try to remove one element and add another (possibly different) to keep maximality.
    This is a 1-swap local search move.
    """
    selected = list(selected)
    best = list(selected)
    best_size = len(selected)
    
    while time.time() < deadline:
        improved = False
        n = len(selected)
        for i in range(n):
            if time.time() >= deadline:
                break
            # Try removing element i
            trial = selected[:i] + selected[i+1:]
            if not trial:
                continue
            trial_sums = _get_sums(trial)
            
            # First check if simply removing works
            if _is_maximal_fast(trial, trial_sums):
                selected = trial
                best = list(selected)
                best_size = len(selected)
                improved = True
                break
            
            # Try swapping: remove i, add some x not in trial
            trial_set = set(trial)
            for x in range(1, N + 1):
                if x in trial_set:
                    continue
                if not _can_add(trial_sums, trial, x):
                    continue
                # Check if trial + [x] is maximal
                new_trial = trial + [x]
                new_sums = set(trial_sums)
                for a in trial:
                    new_sums.add(a + x)
                new_sums.add(2 * x)
                if _is_maximal_fast(new_trial, new_sums):
                    selected = new_trial
                    best = list(selected)
                    best_size = len(selected)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best

def _iterated_local_search(rng, deadline, initial_solution):
    """
    Iterated Local Search: Perturb, Complete, Local Search.
    """
    best = list(initial_solution)
    best_size = len(best)
    
    while time.time() < deadline:
        # Perturb: Remove a random element
        current = list(best)
        if not current:
            break
        
        idx = rng.randint(0, len(current) - 1)
        current.pop(idx)
        
        # Complete to maximal with random order
        order = list(range(1, N + 1))
        rng.shuffle(order)
        current = _complete_to_maximal(current, order)
        
        # Local search
        # Use a short deadline for this step to allow many iterations
        step_deadline = min(deadline, time.time() + 0.3)
        current = _local_search_minimize(current, step_deadline)
        
        if len(current) < best_size:
            best = current
            best_size = len(current)
            
    return best

def _mis_greedy_construct(rng, deadline):
    """
    Maximum Independent Set-inspired greedy construction.
    We view each element x as a vertex. Two elements conflict if they create
    a sum collision. We build a conflict graph implicitly and use a greedy
    independent set approach, but in reverse: we want a small maximal set.
    
    Strategy: Start with empty set. At each step, pick the element that
    "blocks" the most other elements (highest degree in conflict graph)
    among those that can still be added. This tends to create a smaller
    maximal set because each chosen element prevents many others from being added.
    """
    selected = []
    sums = set()
    selected_set = set()
    
    while time.time() < deadline:
        # Find all elements that can be added
        candidates = []
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            if _can_add(sums, selected, x):
                candidates.append(x)
        
        if not candidates:
            break
        
        # For each candidate, compute how many other candidates it would block
        # if we add it
        best_candidate = None
        best_block_count = -1
        
        for x in candidates:
            # If we add x, which other candidates become non-addable?
            new_sums = set(sums)
            for a in selected:
                new_sums.add(a + x)
            new_sums.add(2 * x)
            
            block_count = 0
            for y in candidates:
                if y == x:
                    continue
                if not _can_add(new_sums, selected + [x], y):
                    block_count += 1
            
            if block_count > best_block_count:
                best_block_count = block_count
                best_candidate = x
        
        if best_candidate is None:
            break
        
        # Add the best candidate
        x = best_candidate
        selected.append(x)
        selected_set.add(x)
        for a in selected[:-1]:
            sums.add(a + x)
        sums.add(2 * x)
    
    return selected

def construct():
    deadline = time.time() + 27  # safety margin before 30s kill
    rng = random.Random(12345)
    
    best_maximal = None
    best_size = float('inf')
    
    def update_best(cand):
        nonlocal best_maximal, best_size
        if cand is not None and len(cand) < best_size:
            best_size = len(cand)
            best_maximal = list(cand)
    
    # 1. Get initial solutions from multiple randomized greedy runs
    candidates = []
    # Ascending greedy
    candidates.append(_greedy_construct(list(range(1, N + 1)), N))
    # Descending greedy
    candidates.append(_greedy_construct(list(range(N, 0, -1)), N))
    # Several randomized greedy
    for _ in range(8):
        candidates.append(_randomized_greedy(rng, N))
    
    for cand in candidates:
        update_best(cand)
    
    # 2. MIS-inspired greedy construction
    for _ in range(3):
        mis_cand = _mis_greedy_construct(rng, min(deadline - 5, time.time() + 2))
        update_best(mis_cand)
    
    # 3. Local search on the best candidate
    if best_maximal is not None:
        local_best = _local_search_minimize(best_maximal, min(deadline - 3, time.time() + 3))
        update_best(local_best)
        
        # Also try swap-remove search
        if time.time() < deadline - 2:
            swap_best = _swap_remove_search(best_maximal, min(deadline - 2, time.time() + 2), rng)
            update_best(swap_best)
    
    # 4. Iterated Local Search
    if best_maximal is not None and time.time() < deadline - 5:
        ils_best = _iterated_local_search(rng, deadline - 5, best_maximal)
        update_best(ils_best)
    
    # 5. Try backtracking search for small k
    # Start from a reasonable lower bound
    # For Sidon sets in [1,n], max size is about sqrt(n)
    # Maximal Sidon sets can be smaller
    # Try k from max(5, best_size - 10) down to 5
    min_k_to_try = max(5, best_size - 12)
    max_k_to_try = best_size - 1
    
    for k in range(max_k_to_try, min_k_to_try - 1, -1):
        if time.time() >= deadline - 2:
            break
        result = _backtracking_search(k, deadline - 2, rng)
        if result is not None:
            update_best(result)
            break  # Found one, stop
    
    # 6. More randomized greedy + local search if time remains
    while time.time() < deadline - 1:
        cand = _randomized_greedy(rng, N)
        update_best(cand)
        
        # Local search on this candidate
        local_best = _local_search_minimize(cand, min(deadline - 1, time.time() + 1.5))
        update_best(local_best)
    
    # 7. Final polish: try swap-remove on the best solution
    if best_maximal is not None and time.time() < deadline - 0.5:
        swap_best = _swap_remove_search(best_maximal, deadline - 0.5, rng)
        update_best(swap_best)
    
    # Final validation
    if best_maximal is not None:
        if _is_sidon(best_maximal) and _is_maximal_fast(best_maximal, _get_sums(best_maximal)):
            return sorted(best_maximal)
    
    # Fallback
    return sorted(_greedy_construct(list(range(1, N + 1)), N))
# EVOLVE-BLOCK-END
