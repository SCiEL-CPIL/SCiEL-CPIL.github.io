"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 256, 'maximum_size': 60}

def _compute_cardinality_sums(arr):
    """Compute all subset sums grouped by cardinality for a given array."""
    n = len(arr)
    # sums_by_card[k] = set of all sums of subsets of size k
    sums_by_card = [set() for _ in range(n + 1)]
    sums_by_card[0].add(0)
    
    for i in range(n):
        # Iterate backwards to avoid using the same element twice
        for k in range(i, -1, -1):
            if sums_by_card[k]:
                for s in sums_by_card[k]:
                    sums_by_card[k + 1].add(s + arr[i])
    return sums_by_card

def _is_valid(arr):
    """Check if the set satisfies the constraint: different cardinalities have different sums."""
    n = len(arr)
    sums_by_card = _compute_cardinality_sums(arr)
    
    # For each pair of cardinalities (k1, k2) with k1 != k2, their sum sets must be disjoint
    for k1 in range(n + 1):
        for k2 in range(k1 + 1, n + 1):
            if sums_by_card[k1] and sums_by_card[k2]:
                if sums_by_card[k1] & sums_by_card[k2]:
                    return False
    return True

def _count_collisions(arr):
    """Count the number of cross-cardinality sum collisions."""
    n = len(arr)
    sums_by_card = _compute_cardinality_sums(arr)
    
    collisions = 0
    for k1 in range(n + 1):
        for k2 in range(k1 + 1, n + 1):
            if sums_by_card[k1] and sums_by_card[k2]:
                collisions += len(sums_by_card[k1] & sums_by_card[k2])
    return collisions

def _greedy_construct():
    """
    Greedy construction: try to add the smallest valid integer at each step.
    This tends to keep sums small and separated by cardinality.
    """
    result = []
    for candidate in range(1, 257):
        test_set = result + [candidate]
        if _is_valid(test_set):
            result.append(candidate)
            if len(result) >= P['maximum_size']:
                break
    return result

def _random_search_with_removal(deadline):
    """
    Start with a large random set and iteratively remove elements involved in the most collisions.
    Use a global conflict count to guide removals.
    """
    # Start with a large random subset
    candidates = list(range(1, 257))
    random.shuffle(candidates)
    current_set = candidates[:60]
    best_set = list(current_set)
    best_size = len(current_set)
    
    # Try to improve by removing elements with most collisions
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        if not _is_valid(current_set):
            collisions = _count_collisions(current_set)
            if collisions == 0:
                break
            
            # Find the element involved in the most collisions
            # For each element, count how many collisions it participates in
            element_collisions = defaultdict(int)
            n = len(current_set)
            sums_by_card = _compute_cardinality_sums(current_set)
            
            # For each collision between cardinality k1 and k2, find which elements are involved
            # This is expensive, so we use a heuristic: try removing each element and see if collisions decrease
            min_collisions = collisions
            best_remove = -1
            for idx in range(n):
                test_set = current_set[:idx] + current_set[idx+1:]
                test_collisions = _count_collisions(test_set)
                if test_collisions < min_collisions:
                    min_collisions = test_collisions
                    best_remove = idx
            
            if best_remove >= 0:
                current_set = current_set[:best_remove] + current_set[best_remove+1:]
            else:
                # No single removal helps; try a different random restart
                break
        
        # Try to add elements back
        for candidate in range(1, 257):
            if candidate not in current_set:
                test_set = current_set + [candidate]
                if _is_valid(test_set):
                    current_set = test_set
                    break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
        
        if best_size >= P['maximum_size']:
            break
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline with 10 second safety margin
    
    # First, try greedy construction which is fast and reliable
    greedy_result = _greedy_construct()
    best_result = list(greedy_result)
    best_size = len(best_result)
    
    # If greedy didn't reach maximum, try random search with removal
    if best_size < P['maximum_size'] and time.time() < deadline:
        random.seed(42)
        random_result = _random_search_with_removal(deadline)
        if len(random_result) > best_size:
            best_result = random_result
            best_size = len(best_result)
    
    # Ensure we return a valid list of Python ints
    return [int(x) for x in best_result]
# EVOLVE-BLOCK-END
