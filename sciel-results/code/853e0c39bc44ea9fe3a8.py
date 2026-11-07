"""Search-based maximization of a divisibility-incomparable subset-sum set."""
import time
import random
import sys

DEADLINE_SECONDS = 85.0

def _divides(a, b):
    """True if a divides b (a,b positive ints, a < b)."""
    if a == b:
        return False
    if a <= 0 or b <= 0:
        return False
    return b % a == 0

def _has_divisibility_conflict(sums_set):
    """
    Given a set of positive ints (subset sums), return True if there exist
    two distinct values a < b with a | b.
    """
    arr = sorted(sums_set)
    n = len(arr)
    if n < 2:
        return False
    for i in range(n):
        a = arr[i]
        for j in range(i + 1, n):
            b = arr[j]
            if b % a == 0:
                return True
    return False

def _get_subset_sums(elements):
    """Compute all non-empty subset sums as a set."""
    sums = set()
    n = len(elements)
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    return sums

def _is_valid(elements):
    """Check if the set of elements is valid (no two subset sums divide each other, no duplicate sums)."""
    n = len(elements)
    if n == 0:
        return True
    sums = set()
    # Check for duplicate sums first
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        if s in sums:
            return False
        sums.add(s)
    # Check divisibility
    return not _has_divisibility_conflict(sums)

def _can_add(elements, candidate):
    """Check if adding candidate to elements keeps the set valid."""
    new_elements = elements + [candidate]
    return _is_valid(new_elements)

def _greedy_construct(starting_elements=None, time_limit=5.0):
    """
    Greedy construction: start with some elements and try to add more.
    Returns a valid set.
    """
    if starting_elements is None:
        starting_elements = []
    
    start_time = time.time()
    best = list(starting_elements)
    
    # Try to add elements greedily
    candidates = list(range(2, 257))
    random.shuffle(candidates)
    
    current = list(starting_elements)
    
    for cand in candidates:
        if time.time() - start_time > time_limit:
            break
        if cand in current:
            continue
        if _can_add(current, cand):
            current.append(cand)
    
    if len(current) > len(best):
        best = current
    
    return best

def _local_search(elements, time_limit=3.0):
    """
    Try to improve the set by removing one element and adding another.
    """
    start_time = time.time()
    current = list(elements)
    best = list(current)
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try removing each element and adding a new one
        for i in range(len(current)):
            if time.time() - start_time > time_limit:
                break
            remaining = current[:i] + current[i+1:]
            # Try to add random candidates
            candidates = [x for x in range(2, 257) if x not in remaining]
            random.shuffle(candidates)
            
            for cand in candidates[:20]:  # Limit candidates tried
                if time.time() - start_time > time_limit:
                    break
                new_set = remaining + [cand]
                if _is_valid(new_set):
                    current = new_set
                    improved = True
                    if len(current) > len(best):
                        best = list(current)
                    break
        if not improved:
            break
    
    return best

def construct():
    """
    Main construction function. Uses randomized greedy search with local improvement.
    """
    start_time = time.time()
    best_result = [31, 32, 34]  # Seed result
    
    # Try multiple random restarts
    for attempt in range(50):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
        
        # Random starting point
        if attempt == 0:
            start = []
        else:
            # Start with a random small set
            size = random.randint(1, 5)
            start = random.sample(range(2, 257), size)
        
        # Greedy construction
        result = _greedy_construct(start, time_limit=2.0)
        
        if len(result) > len(best_result):
            best_result = result
        
        # Local search to improve
        if len(result) > len(best_result) - 2:  # Only if close to best
            improved = _local_search(result, time_limit=1.0)
            if len(improved) > len(best_result):
                best_result = improved
        
        # Check deadline
        if time.time() - start_time > DEADLINE_SECONDS - 3:
            break
    
    # Final verification
    if _is_valid(best_result):
        return list(best_result)
    else:
        return [31, 32, 34]

# EVOLVE-BLOCK-END
