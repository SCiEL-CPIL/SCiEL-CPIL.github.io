"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import random
import time

def _is_sidon_and_maximal(selected, N):
    """Check if selected is a valid Sidon set and inclusion-maximal in [1,N]."""
    if len(selected) == 0:
        return False
    selected_set = set(selected)
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            b = selected[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    # Check maximality: no missing element can be added
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        added_sums = set()
        for a in selected:
            s1 = x + a
            if s1 in sums or s1 in added_sums:
                break
            added_sums.add(s1)
        else:
            s2 = 2 * x
            if s2 not in sums and s2 not in added_sums:
                # x could be added, so not maximal
                return False
    return True

def _greedy_with_order(order, N):
    """Greedy construction using a specific element ordering."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def construct():
    deadline = time.time() + 55.0  # 55s safety margin
    best_size = N + 1
    best_set = None
    
    # Try ascending greedy first (baseline)
    baseline = _greedy_with_order(list(range(1, N + 1)), N)
    if _is_sidon_and_maximal(baseline, N):
        best_size = len(baseline)
        best_set = baseline
    
    # Random restarts with different orderings
    rng = random.Random(42)
    for attempt in range(10000):
        if time.time() > deadline:
            break
        
        # Generate a random ordering with bias
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        candidate = _greedy_with_order(order, N)
        
        if len(candidate) < best_size and _is_sidon_and_maximal(candidate, N):
            best_size = len(candidate)
            best_set = candidate
            if best_size <= 16:  # Good enough, early stop
                break
    
    if best_set is None:
        # Fallback to baseline
        best_set = _greedy_with_order(list(range(1, N + 1)), N)
    
    return best_set
# EVOLVE-BLOCK-END
