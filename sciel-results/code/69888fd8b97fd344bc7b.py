"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

N = 256
DEADLINE_SECONDS = 55.0  # safety margin under 60s wall-clock

# EVOLVE-BLOCK-START

def _build_sums(selected):
    """Return the set of all unordered pair sums a+b with a<=b from selected."""
    sums = set()
    for i, a in enumerate(selected):
        sums.add(2 * a)
        for b in selected[i + 1:]:
            sums.add(a + b)
    return sums


def _conflicts_valid(candidate, selected, sums):
    """Check if adding candidate to selected keeps the Sidon property.
    Returns True if valid, False otherwise."""
    # Check 2*candidate and candidate + a for all a in selected
    # against existing sums and against each other
    new_sums = [2 * candidate]
    for a in selected:
        new_sums.append(candidate + a)
    # Check for duplicates within new_sums
    if len(set(new_sums)) != len(new_sums):
        return False
    # Check against existing sums
    for s in new_sums:
        if s in sums:
            return False
    return True


def _greedy_maximal_sidon(order):
    """Greedy construction: iterate through order, add element if it doesn't conflict.
    Returns (selected, sums) where selected is the maximal Sidon set."""
    selected = []
    sums = set()
    for x in order:
        if _conflicts_valid(x, selected, sums):
            selected.append(x)
            sums.add(2 * x)
            for a in selected[:-1]:
                sums.add(x + a)
    return selected, sums


def _is_maximal(selected, sums):
    """Check if no missing element can be added while preserving Sidon property."""
    selected_set = set(selected)
    for x in range(1, N + 1):
        if x not in selected_set:
            if _conflicts_valid(x, selected, sums):
                return False
    return True


def _try_remove(selected, sums):
    """Try to remove elements one by one to reduce size while maintaining maximality.
    Returns a smaller maximal Sidon set if found, else None."""
    current = list(selected)
    current_sums = _build_sums(current)
    
    # Try removing each element, check if still maximal
    for i in range(len(current)):
        candidate = current[i]
        new_list = current[:i] + current[i+1:]
        if not new_list:
            continue
        new_sums = _build_sums(new_list)
        # Check maximality of the reduced set
        if _is_maximal(new_list, new_sums):
            return new_list, new_sums
    
    return None


def _local_search(selected, sums, max_iters=200):
    """Local search: try removing elements and re-adding to find smaller maximal sets."""
    current = list(selected)
    current_sums = _build_sums(current)
    best = (len(current), current, current_sums)
    
    for _ in range(max_iters):
        improved = False
        # Try removing each element
        for i in range(len(current)):
            candidate = current[i]
            new_list = current[:i] + current[i+1:]
            if not new_list:
                continue
            new_sums = _build_sums(new_list)
            if _is_maximal(new_list, new_sums):
                if len(new_list) < best[0]:
                    best = (len(new_list), new_list, new_sums)
                    current = new_list
                    current_sums = new_sums
                    improved = True
                break
        if not improved:
            break
    
    return best[1], best[2]


def construct():
    """Construct a small inclusion-maximal Sidon set in [1, N].
    Uses randomized greedy with local search to minimize cardinality."""
    start_time = time.time()
    
    best_selected = None
    best_sums = None
    best_size = N + 1
    
    # First, run a deterministic greedy to get a baseline
    order = list(range(1, N + 1))
    selected, sums = _greedy_maximal_sidon(order)
    best_selected = list(selected)
    best_sums = sums
    best_size = len(selected)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS:
        return best_selected
    
    # Run randomized greedy with different orderings
    rng = random.Random(42)
    
    # Try various random orderings
    while time.time() - start_time < DEADLINE_SECONDS - 2.0:
        # Create a random permutation
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        selected, sums = _greedy_maximal_sidon(order)
        
        if len(selected) < best_size:
            best_selected = list(selected)
            best_sums = sums
            best_size = len(selected)
        
        # Try local search to reduce size further
        if len(selected) <= best_size:
            reduced, reduced_sums = _local_search(selected, sums, max_iters=50)
            if len(reduced) < best_size:
                best_selected = reduced
                best_sums = reduced_sums
                best_size = len(reduced)
    
    # Final local search on the best found so far
    if time.time() - start_time < DEADLINE_SECONDS - 1.0:
        reduced, reduced_sums = _local_search(best_selected, best_sums, max_iters=100)
        if len(reduced) < best_size:
            best_selected = reduced
            best_sums = reduced_sums
            best_size = len(reduced)
    
    return best_selected

# EVOLVE-BLOCK-END
