"""Model-and-optimize approach for minimum-size maximal Sidon sets in [1,256]."""
import time
import random
import sys

N = 256
DEADLINE = 58.0  # safety margin under 60s wall-clock

# EVOLVE-BLOCK-START

def _sidon_sums(S):
    """Return set of all unordered sums a+b with a<=b from S."""
    sums = set()
    for i, a in enumerate(S):
        for b in S[i:]:
            sums.add(a + b)
    return sums

def _is_sidon(S):
    """Check if S is a Sidon set: all unordered sums a+b with a<=b are distinct."""
    sums = set()
    for i, a in enumerate(S):
        for b in S[i:]:
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_maximal_sidon(start=1, end=None, step=1, skip_pattern=None):
    """Build a maximal Sidon set greedily by scanning and adding when possible."""
    if end is None:
        end = N
    if skip_pattern is None:
        skip_pattern = set()
    selected = []
    sums = set()
    x = start
    while x <= end:
        if x not in skip_pattern:
            new_sums = [x + a for a in selected] + [2 * x]
            if len(set(new_sums)) == len(new_sums) and not (sums & set(new_sums)):
                selected.append(x)
                sums.update(new_sums)
        x += step
    # Ensure maximality: add remaining elements that can be added
    S_set = set(selected)
    for y in range(1, end + 1):
        if y in S_set:
            continue
        new_sums = [y + a for a in selected] + [2 * y]
        if len(set(new_sums)) == len(new_sums) and not (sums & set(new_sums)):
            selected.append(y)
            sums.update(new_sums)
            S_set.add(y)
    return selected

def _addable_elements(S, sums, universe=None):
    """Return set of elements in [1,N] not in S that could be added while keeping Sidon."""
    if universe is None:
        universe = range(1, N + 1)
    S_set = set(S)
    addable = []
    for y in universe:
        if y in S_set:
            continue
        new_sums = [y + a for a in S] + [2 * y]
        if len(set(new_sums)) == len(new_sums) and not (sums & set(new_sums)):
            addable.append(y)
    return addable

def _is_maximal(S):
    """Check if S is a maximal Sidon set (no element from [1,N] can be added)."""
    sums = _sidon_sums(S)
    addable = _addable_elements(S, sums)
    return len(addable) == 0, addable

def _try_remove_and_repair(S, t_start):
    """
    Try to remove an element from S and repair maximality.
    Returns a smaller maximal Sidon set if found, else None.
    """
    S_list = list(S)
    n = len(S_list)
    
    for idx in range(n):
        if time.time() - t_start > DEADLINE - 2.0:
            break
        # Try removing S_list[idx]
        removed = S_list[idx]
        S_trial = [x for j, x in enumerate(S_list) if j != idx]
        sums_trial = _sidon_sums(S_trial)
        
        # S_trial is still Sidon (subset). Check maximality.
        addable = _addable_elements(S_trial, sums_trial)
        if len(addable) == 0:
            # Great! Removing it keeps maximality. Size decreased.
            return S_trial
    
    return None

def _local_search_minimize(S, t_start):
    """
    Local search: repeatedly try to find a removal that preserves maximality.
    Also try swaps to enable future removals.
    """
    best = list(S)
    best_size = len(best)
    
    while time.time() - t_start < DEADLINE - 2.0:
        improved = False
        
        # Try each element for removal
        for i in range(len(best)):
            if time.time() - t_start > DEADLINE - 2.0:
                break
            removed = best[i]
            S_trial = [x for j, x in enumerate(best) if j != i]
            sums_trial = _sidon_sums(S_trial)
            addable = _addable_elements(S_trial, sums_trial)
            
            if len(addable) == 0:
                # Direct removal works!
                best = S_trial
                best_size -= 1
                improved = True
                break
            else:
                # Try swap: remove best[i], add some y from addable
                for y in random.sample(addable, min(5, len(addable))):
                    if time.time() - t_start > DEADLINE - 2.0:
                        break
                    S_swap = S_trial + [y]
                    S_swap.sort()
                    sums_swap = _sidon_sums(S_swap)
                    if _is_sidon(S_swap):
                        # Try to remove some other element from S_swap
                        for j in range(len(S_swap)):
                            if time.time() - t_start > DEADLINE - 2.0:
                                break
                            S_trial2 = [x for k, x in enumerate(S_swap) if k != j]
                            sums_trial2 = _sidon_sums(S_trial2)
                            addable2 = _addable_elements(S_trial2, sums_trial2)
                            if len(addable2) == 0:
                                # Swap + removal reduces size by 1
                                best = S_trial2
                                best_size -= 1
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            # No improvement possible with current local moves.
            # Try perturbation
            if len(best) > 1 and random.random() < 0.3:
                idx = random.randint(0, len(best) - 1)
                S_pert = [x for j, x in enumerate(best) if j != idx]
                sums_pert = _sidon_sums(S_pert)
                addable_pert = _addable_elements(S_pert, sums_pert)
                if len(addable_pert) > 0:
                    # Add a random element to restore some structure
                    y = random.choice(addable_pert)
                    S_pert.append(y)
                    S_pert.sort()
                    if _is_sidon(S_pert):
                        best = S_pert
                        best_size = len(best)
                        continue
            break
    
    return best

def _construct_via_repair():
    """
    Build a maximal Sidon set, then iteratively minimize its size
    by finding elements whose removal preserves maximality.
    """
    t_start = time.time()
    
    # Try multiple greedy constructions with different parameters
    candidates = []
    
    # Standard greedy
    S1 = _greedy_maximal_sidon()
    candidates.append(S1)
    
    # Greedy with different start
    S2 = _greedy_maximal_sidon(start=2)
    candidates.append(S2)
    
    # Greedy with step 2, then fill
    S3 = _greedy_maximal_sidon(start=1, step=2)
    candidates.append(S3)
    
    # Greedy with random skips
    skip1 = set(random.sample(range(1, N+1), 10))
    S4 = _greedy_maximal_sidon(skip_pattern=skip1)
    candidates.append(S4)
    
    skip2 = set(random.sample(range(1, N+1), 20))
    S5 = _greedy_maximal_sidon(skip_pattern=skip2)
    candidates.append(S5)
    
    # Pick the smallest candidate
    best = min(candidates, key=len)
    print(f"Best initial candidate size: {len(best)}", file=sys.stderr)
    
    # Iteratively try to remove elements
    improved = True
    while improved and time.time() - t_start < DEADLINE - 5.0:
        improved = False
        for i in range(len(best)):
            if time.time() - t_start > DEADLINE - 5.0:
                break
            removed = best[i]
            S_trial = [x for j, x in enumerate(best) if j != i]
            sums_trial = _sidon_sums(S_trial)
            addable = _addable_elements(S_trial, sums_trial)
            if len(addable) == 0:
                best = S_trial
                improved = True
                break
        # If no single removal works, try more advanced local search
        if not improved:
            S_new = _local_search_minimize(best, t_start)
            if len(S_new) < len(best):
                best = S_new
                improved = True
    
    return best

def construct():
    """Return a minimum-size maximal Sidon set in [1,256]."""
    t_start = time.time()
    
    # Use the model-and-optimize approach
    result = _construct_via_repair()
    
    # Verify the result
    if not result or not _is_sidon(result):
        # Fallback to greedy
        result = _greedy_maximal_sidon()
    
    return result

# EVOLVE-BLOCK-END
