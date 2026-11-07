"""Multi-phase search: biased greedy + local hill-climbing with swap/insert moves."""
# EVOLVE-BLOCK-START
import time
import random
import math

# Deadline guard: 50s wall-clock (10s safety margin under 60s limit)
_DEADLINE = 50.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _check_no_divisibility(sums):
    """Check that no two distinct values in sums divide one another."""
    if not sums:
        return True
    
    sum_set = set(sums)
    max_sum = max(sum_set)
    
    for s in sum_set:
        multiple = s * 2
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += s
            
    return True

def _subset_sums_incremental(current_sums, new_element):
    """Given a set of existing non-empty subset sums, add a new element."""
    new_sums = set(current_sums)
    new_sums.add(new_element)
    for s in current_sums:
        new_sums.add(s + new_element)
    return new_sums

def _remove_element_from_sums(current_sums, elements, idx):
    """Remove element at idx from the set and recompute sums.
    elements is a list, current_sums is the full set of non-empty subset sums."""
    n = len(elements)
    elem = elements[idx]
    # Build new element list without idx
    new_elements = [elements[i] for i in range(n) if i != idx]
    # Recompute all subset sums
    new_sums = set()
    m = len(new_elements)
    for mask in range(1, 1 << m):
        s = 0
        for i in range(m):
            if mask & (1 << i):
                s += new_elements[i]
        new_sums.add(s)
    return new_elements, new_sums

def _validate_full(elements):
    """Full validation: distinct, in range, all non-empty subset sums no divisibility."""
    if not elements:
        return True
    if len(elements) != len(set(elements)):
        return False
    for x in elements:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x < 1 or x > 256:
            return False
    n = len(elements)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    if not _check_no_divisibility(sums):
        return False
    return True

def _greedy_construct(candidates, deadline_check=None, bias_large=True):
    """Greedy: try adding candidates in given order, keeping only valid additions."""
    chosen = []
    current_sums = set()
    chosen_set = set()
    
    for x in candidates:
        if deadline_check and deadline_check() < 1.0:
            break
        if x in chosen_set:
            continue
        new_sums = _subset_sums_incremental(current_sums, x)
        if _check_no_divisibility(new_sums):
            chosen.append(x)
            chosen_set.add(x)
            current_sums = new_sums
    
    return chosen

def _local_search(elements, pool, deadline_check=None, max_iterations=200):
    """Local hill-climbing: try swap and insert moves to improve set size."""
    if not elements:
        return elements
    
    chosen = list(elements)
    chosen_set = set(chosen)
    
    # Compute current sums
    n = len(chosen)
    current_sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += chosen[i]
        current_sums.add(s)
    
    best_len = len(chosen)
    best_chosen = list(chosen)
    
    for iteration in range(max_iterations):
        if deadline_check and deadline_check() < 2.0:
            break
        
        improved = False
        
        # Try insert moves: add a new element from pool
        for x in pool:
            if x in chosen_set:
                continue
            new_sums = _subset_sums_incremental(current_sums, x)
            if _check_no_divisibility(new_sums):
                chosen.append(x)
                chosen_set.add(x)
                current_sums = new_sums
                improved = True
                break
        
        if improved:
            best_len = len(chosen)
            best_chosen = list(chosen)
            continue
        
        # Try swap moves: remove one element, add another
        swapped = False
        for i in range(len(chosen)):
            if deadline_check and deadline_check() < 2.0:
                break
            elem_to_remove = chosen[i]
            new_elements, new_sums = _remove_element_from_sums(current_sums, chosen, i)
            new_elements_set = set(new_elements)
            
            # Try to find a replacement that increases size or maintains
            for x in pool:
                if x in new_elements_set:
                    continue
                # After removing, we have len(chosen)-1 elements. Adding x gives len(chosen) elements.
                # But we want to potentially get more, so also check if we can then add another.
                test_sums = _subset_sums_incremental(new_sums, x)
                if _check_no_divisibility(test_sums):
                    # Valid swap: replace chosen[i] with x
                    chosen[i] = x
                    chosen_set.discard(elem_to_remove)
                    chosen_set.add(x)
                    current_sums = test_sums
                    swapped = True
                    break
            
            if swapped:
                break
        
        if not swapped:
            # Try double-swap: remove two, add two (to potentially increase size by 0 but rearrange)
            # Or just break if no improvement possible
            break
    
    return best_chosen

def _biased_order(pool, rng):
    """Generate a candidate order biased toward large numbers and primes."""
    # Score each number: prefer large, prefer prime
    def score(x):
        s = x * 0.5  # Large number bias
        # Simple primality test for small numbers
        if x > 1:
            is_prime = True
            for d in range(2, int(math.isqrt(x)) + 1):
                if x % d == 0:
                    is_prime = False
                    break
            if is_prime:
                s += 20
        return s
    
    # Sort by score with some randomness
    scored = [(score(x), rng.random(), x) for x in pool]
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return [x for _, _, x in scored]

def construct():
    global _START
    _START = time.time()
    
    pool = list(range(1, 257))
    
    best = []
    best_len = 0
    
    rng = random.Random(42)
    
    # Phase 1: Deterministic greedy with large-first ordering
    order1 = sorted(pool, reverse=True)
    result1 = _greedy_construct(order1, deadline_check=_remaining)
    if len(result1) > best_len:
        best = result1
        best_len = len(result1)
    
    # Phase 2: Biased greedy
    if _remaining() > 5.0:
        biased_order = _biased_order(pool, rng)
        result2 = _greedy_construct(biased_order, deadline_check=_remaining)
        if len(result2) > best_len:
            best = result2
            best_len = len(result2)
    
    # Phase 3: Randomized greedy restarts + local search
    while _remaining() > 3.0:
        random_pool = list(pool)
        rng.shuffle(random_pool)
        
        result = _greedy_construct(random_pool, deadline_check=_remaining)
        
        # Apply local search to improve
        if len(result) >= best_len - 1 and _remaining() > 4.0:
            improved = _local_search(result, pool, deadline_check=_remaining, max_iterations=50)
            if len(improved) > len(result):
                result = improved
        
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Final local search on best
    if best and _remaining() > 5.0:
        final = _local_search(best, pool, deadline_check=_remaining, max_iterations=100)
        if len(final) > best_len:
            best = final
            best_len = len(final)
    
    # Final validation
    if best and not _validate_full(best):
        best = [128, 192, 224]
    
    return [int(x) for x in best]

# EVOLVE-BLOCK-END
