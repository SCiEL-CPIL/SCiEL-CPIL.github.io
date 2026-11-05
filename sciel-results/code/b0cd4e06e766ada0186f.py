"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256
import time
import random
import sys
import numpy as np

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
    Tries to remove elements one by one, with incremental sum updates.
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
            removed = selected[i]
            trial = selected[:i] + selected[i+1:]
            if not trial:
                continue
            
            # Compute trial sums efficiently
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

def _iterated_local_search(rng, deadline, initial_solution):
    """
    Iterated Local Search: Perturb (remove 1-3 random elements), Complete, Local Search.
    """
    best = list(initial_solution)
    best_size = len(best)
    
    while time.time() < deadline:
        # Perturb: Remove 1 to 3 random elements
        current = list(best)
        if not current:
            break
        
        n_remove = rng.randint(1, min(3, len(current)))
        indices = rng.sample(range(len(current)), n_remove)
        for idx in sorted(indices, reverse=True):
            current.pop(idx)
        
        # Complete to maximal
        current = _complete_to_maximal(current)
        
        # Local search
        step_deadline = min(deadline, time.time() + 0.3)
        current = _local_search_minimize(current, step_deadline)
        
        if len(current) < best_size:
            best = current
            best_size = len(current)
            
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
    Uses pruning: if not enough elements remain to reach size k, backtrack.
    """
    def backtrack(current, current_sums, start_idx):
        if len(current) == k:
            # Check if maximal
            if _is_maximal_fast(current, current_sums):
                return list(current)
            return None
        
        if time.time() >= deadline:
            return None
        
        remaining_needed = k - len(current)
        # Pruning: not enough elements left
        if N - start_idx + 1 < remaining_needed:
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

def _vectorized_blocking_count(selected, sums, candidates):
    """
    For each candidate x, count how many currently-addable elements it would block.
    An element y is 'addable' if it can currently be added to selected.
    x blocks y if adding x first makes y no longer addable.
    x blocks y if:
      - x+y is in sums (so y can't be added after x), OR
      - 2*y is in sums (already blocked, so y wasn't addable anyway - don't count)
      - 2*x is in sums (x can't be added at all - shouldn't be in candidates)
    
    Actually, the blocking count for x is: number of y in candidates such that
    y is currently addable, but after adding x, y is NOT addable.
    
    y is currently addable if: for all a in selected, a+y not in sums, and 2y not in sums.
    After adding x, y is not addable if: (x+y in sums) OR (for some a in selected, a+y in sums) OR (2y in sums).
    The last two conditions are the same as y being currently addable being false.
    So y is blocked by x iff y is currently addable AND (x+y in sums OR 2*x in sums... wait no).
    
    Let me reconsider. y is addable to current set S iff:
      1. 2y not in sums(S)
      2. For all a in S: a+y not in sums(S)
    
    After adding x to S, y is addable to S∪{x} iff:
      1. 2y not in sums(S∪{x})
      2. For all a in S∪{x}: a+y not in sums(S∪{x})
    
    sums(S∪{x}) = sums(S) ∪ {a+x for a in S} ∪ {2x}
    
    So y is NOT addable to S∪{x} iff:
      - 2y in sums(S∪{x}), OR
      - exists a in S∪{x} such that a+y in sums(S∪{x})
    
    If y IS addable to S, then 2y not in sums(S) and for all a in S, a+y not in sums(S).
    
    y becomes NOT addable after adding x iff:
      - 2y in {a+x for a in S} ∪ {2x}, i.e., 2y = a+x for some a in S, or 2y = 2x (i.e., y=x)
      - OR exists a in S∪{x} such that a+y in sums(S∪{x})
        = exists a in S such that a+y in sums(S∪{x}), OR x+y in sums(S∪{x})
        = exists a in S such that a+y in sums(S) ∪ {b+x for b in S} ∪ {2x}, OR x+y in sums(S) ∪ {b+x for b in S} ∪ {2x}
    
    This is getting complex. Let me use a simpler approach:
    For each candidate x, compute the set of elements it would block.
    """
    selected_set = set(selected)
    n_cands = len(candidates)
    if n_cands == 0:
        return np.zeros(0, dtype=np.int32)
    
    # First, determine which candidates are currently addable
    addable = np.zeros(n_cands, dtype=bool)
    for i, x in enumerate(candidates):
        if x in selected_set:
            continue
        can = True
        for a in selected:
            if a + x in sums:
                can = False
                break
        if can and 2 * x in sums:
            can = False
        addable[i] = can
    
    # For each addable candidate x, count how many other addable candidates y it blocks
    blocking = np.zeros(n_cands, dtype=np.int32)
    
    for i in range(n_cands):
        x = candidates[i]
        if not addable[i]:
            continue
        
        # After adding x, what new sums are created?
        new_sums = set()
        for a in selected:
            new_sums.add(a + x)
        new_sums.add(2 * x)
        
        count = 0
        for j in range(n_cands):
            if i == j or not addable[j]:
                continue
            y = candidates[j]
            # Check if y is blocked by x
            # y is blocked if: 2y in new_sums, or (x+y) in (sums ∪ new_sums), or exists a in selected with a+y in new_sums
            # Since y is currently addable, 2y not in sums and a+y not in sums for all a in selected.
            # So y is blocked iff:
            #   2y in new_sums, OR
            #   x+y in sums ∪ new_sums, OR
            #   exists a in selected such that a+y in new_sums
            
            blocked = False
            if 2 * y in new_sums:
                blocked = True
            elif (x + y) in sums or (x + y) in new_sums:
                blocked = True
            else:
                for a in selected:
                    if a + y in new_sums:
                        blocked = True
                        break
            
            if blocked:
                count += 1
        
        blocking[i] = count
    
    return blocking

def _greedy_saturation_with_blocking(rng, deadline, initial_solution=None):
    """
    Greedy saturation that prioritizes elements with high blocking counts.
    Start from a Sidon set, then greedily add elements that block the most
    currently-addable elements.
    """
    if initial_solution is not None:
        selected = list(initial_solution)
    else:
        selected = []
    
    sums = _get_sums(selected)
    selected_set = set(selected)
    
    while time.time() < deadline:
        # Find all candidates that can be added
        candidates = [x for x in range(1, N + 1) if x not in selected_set]
        if not candidates:
            break
        
        # Compute blocking counts
        blocking = _vectorized_blocking_count(selected, sums, candidates)
        
        # Find the candidate with maximum blocking count
        max_idx = np.argmax(blocking)
        if blocking[max_idx] == 0:
            # No element blocks others; just add the first available
            x = candidates[max_idx]
        else:
            x = candidates[max_idx]
        
        # Add x
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
        if len(cand) < best_size:
            best_size = len(cand)
            best_maximal = list(cand)
    
    # 2. Local search on the best candidate
    if best_maximal is not None:
        local_best = _local_search_minimize(best_maximal, min(deadline, time.time() + 5))
        if len(local_best) < best_size:
            best_size = len(local_best)
            best_maximal = local_best
            
    # 3. Iterated Local Search
    if best_maximal is not None and time.time() < deadline - 8:
        ils_best = _iterated_local_search(rng, deadline - 8, best_maximal)
        if len(ils_best) < best_size:
            best_size = len(ils_best)
            best_maximal = ils_best

    # 4. Try backtracking search for small k
    min_k_to_try = max(5, best_size - 12)
    max_k_to_try = best_size - 1
    
    for k in range(max_k_to_try, min_k_to_try - 1, -1):
        if time.time() >= deadline - 3:
            break
        result = _backtracking_search(k, deadline - 3, rng)
        if result is not None:
            if len(result) < best_size:
                best_size = len(result)
                best_maximal = result
            break  # Found one, stop
    
    # 5. More randomized greedy + local search if time remains
    while time.time() < deadline - 1:
        cand = _randomized_greedy(rng, N)
        if len(cand) < best_size:
            best_size = len(cand)
            best_maximal = list(cand)
        
        # Local search on this candidate
        local_best = _local_search_minimize(cand, min(deadline - 1, time.time() + 1.5))
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
