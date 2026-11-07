"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

MAX_SUM = 256 * 30  # Maximum possible subset sum for 30 elements each <= 256
MAX_ELEMENTS = 30

def _precompute_divisibility_forbidden():
    """Precompute for each sum s, the set of multiples of s up to MAX_SUM."""
    forbidden = [set() for _ in range(MAX_SUM + 1)]
    for s in range(1, MAX_SUM + 1):
        for k in range(2, MAX_SUM // s + 1):
            forbidden[s].add(s * k)
    return forbidden

def _get_subset_sums(elements):
    """Get all nonempty subset sums as a set using iterative method."""
    sums = set()
    for e in elements:
        new_sums = set()
        for s in sums:
            new_sums.add(s + e)
        sums = sums | new_sums | {e}
    return sums

def _check_divisibility(sums, forbidden):
    """Check if no two distinct sums in the set divide each other."""
    sorted_sums = sorted(sums)
    for i, a in enumerate(sorted_sums):
        if a > MAX_SUM:
            continue
        for j in range(i + 1, len(sorted_sums)):
            b = sorted_sums[j]
            if b > MAX_SUM:
                break
            if b in forbidden[a]:
                return False
    return True

def _is_valid_set(elements, forbidden):
    """Check if a set of integers has the required properties."""
    if not elements:
        return True
    sums = _get_subset_sums(elements)
    return _check_divisibility(sums, forbidden)

def _check_addition_fast(current_sums, new_elem, forbidden):
    """
    Check if adding new_elem to current set maintains validity.
    Returns new sums set if valid, None otherwise.
    """
    # Compute new sums: old_sums ∪ {new_elem} ∪ {s + new_elem for s in old_sums}
    new_sums = set(current_sums)
    new_sums.add(new_elem)
    for s in current_sums:
        new_sums.add(s + new_elem)
    
    # Check divisibility among all pairs in new_sums
    sorted_sums = sorted(new_sums)
    for i, a in enumerate(sorted_sums):
        if a > MAX_SUM:
            continue
        for j in range(i + 1, len(sorted_sums)):
            b = sorted_sums[j]
            if b > MAX_SUM:
                break
            if b in forbidden[a]:
                return None
    return new_sums

def _greedy_grow(current_elements, current_sums, forbidden, deadline):
    """
    Greedily grow the set by trying all candidate values in random order.
    Returns the largest valid set found.
    """
    best = current_elements[:]
    best_sums = current_sums
    
    # Try to add elements one by one
    for _ in range(30 - len(current_elements)):
        if time.time() > deadline:
            break
        
        # Collect all valid candidates
        valid_candidates = []
        for val in range(1, 257):
            if val in current_elements:
                continue
            new_sums = _check_addition_fast(current_sums, val, forbidden)
            if new_sums is not None:
                valid_candidates.append((val, new_sums))
        
        if not valid_candidates:
            break
        
        # Pick randomly from valid candidates (or greedily pick smallest/largest)
        # Use random selection for diversity
        val, new_sums = random.choice(valid_candidates)
        current_elements = current_elements + [val]
        current_sums = new_sums
        best = current_elements[:]
        best_sums = current_sums
    
    return best

def _try_strategy(deadline, forbidden, seed=None):
    """Try a random greedy strategy."""
    if seed is not None:
        random.seed(seed)
    
    best = []
    
    # Start with a random initial element
    start_val = random.randint(2, 200)
    initial_elements = [start_val]
    initial_sums = {start_val}
    
    result = _greedy_grow(initial_elements, initial_sums, forbidden, deadline)
    if len(result) > len(best):
        best = result
    
    return best

def _backtrack_limited(current_elements, current_sums, forbidden, deadline, depth_limit=3):
    """
    Limited backtracking to explore alternatives at each level.
    """
    best = current_elements[:]
    best_size = len(current_elements)
    
    if depth_limit <= 0:
        return best
    
    # Try each valid candidate
    candidates = []
    for val in range(1, 257):
        if val in current_elements:
            continue
        new_sums = _check_addition_fast(current_sums, val, forbidden)
        if new_sums is not None:
            candidates.append((val, new_sums))
    
    if not candidates:
        return best
    
    # Limit number of candidates to explore
    max_candidates = min(len(candidates), 10)
    # Randomly select which candidates to explore
    selected = random.sample(candidates, max_candidates)
    
    for val, new_sums in selected:
        if time.time() > deadline:
            break
        
        new_elements = current_elements + [val]
        result = _backtrack_limited(new_elements, new_sums, forbidden, deadline, depth_limit - 1)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    # Precompute forbidden divisibility pairs
    forbidden = _precompute_divisibility_forbidden()
    
    best = []
    
    # Strategy 1: Multiple random greedy attempts
    for attempt in range(50):
        if time.time() > deadline:
            break
        result = _try_strategy(deadline, forbidden, seed=attempt)
        if len(result) > len(best):
            best = result
        if len(best) >= 8:
            break
    
    # Strategy 2: Limited backtracking from good starting points
    if len(best) < 8:
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Start from a random small set
            size = random.randint(1, 3)
            init_elems = random.sample(range(2, 257), size)
            init_elems.sort()
            init_sums = _get_subset_sums(init_elems)
            
            if not _check_divisibility(init_sums, forbidden):
                continue
            
            result = _backtrack_limited(init_elems, init_sums, forbidden, deadline, depth_limit=4)
            if len(result) > len(best):
                best = result
    
    # Ensure we return a list of Python ints
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
