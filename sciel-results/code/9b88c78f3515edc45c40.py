import time
import random
import math

# Deadline guard: 80s wall-clock (10s safety margin under 90s limit)
_DEADLINE = 80.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _get_sums_bitset(elements):
    """
    Compute the bitset of all non-empty subset sums.
    Bit i is 1 if sum i is achievable.
    """
    sums = 0
    for x in elements:
        sums |= (sums << x)
        sums |= (1 << x)
    # Clear the 0th bit (empty subset)
    sums &= ~(1 << 0)
    return sums

def _check_no_divisibility_bitset(sums_bitset):
    """
    Check if no two distinct values in the bitset divide one another.
    sums_bitset: int where bit i is set if i is a subset sum.
    Returns True if valid, False otherwise.
    """
    if sums_bitset == 0:
        return True
    
    max_sum = sums_bitset.bit_length() - 1
    
    # Extract sums to a list for easier iteration if needed, 
    # but we can also just iterate through bits.
    # For efficiency, let's get the list of sums.
    sums_list = []
    temp = sums_bitset
    while temp:
        # Get index of lowest set bit
        # Python 3.11+ has int.bit_count, but for lowest set bit:
        # (temp & -temp).bit_length() - 1
        lsb = temp & -temp
        idx = lsb.bit_length() - 1
        sums_list.append(idx)
        temp ^= lsb
        
    # Check divisibility
    # For each s in sums_list, check if any multiple 2s, 3s... is in the set.
    # We use the bitset for O(1) lookup.
    for s in sums_list:
        multiple = s * 2
        while multiple <= max_sum:
            if (sums_bitset >> multiple) & 1:
                return False
            multiple += s
            
    return True

def _is_valid_addition(current_sums_bitset, x, n_current):
    """
    Check if adding x to a set with current_sums_bitset (size n_current) is valid.
    Returns True if valid.
    """
    # New sums = old sums | (old sums shifted by x) | (1 << x)
    new_sums = current_sums_bitset | (current_sums_bitset << x) | (1 << x)
    
    # Check distinctness:
    # Expected count is 2^(n_current+1) - 1
    expected_count = (1 << (n_current + 1)) - 1
    if new_sums.bit_count() != expected_count:
        return False
        
    # Check divisibility
    return _check_no_divisibility_bitset(new_sums)

def _validate_full(elements):
    """
    Full validation for the final result.
    """
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
    sums_bitset = _get_sums_bitset(elements)
    
    # Check distinctness of sums
    if sums_bitset.bit_count() != (1 << n) - 1:
        return False
        
    # Check divisibility
    if not _check_no_divisibility_bitset(sums_bitset):
        return False
        
    return True

def _greedy_construct(candidates, deadline_check=None):
    """
    Greedy construction using bitset validation.
    """
    chosen = []
    current_sums_bitset = 0
    n_chosen = 0
    
    for x in candidates:
        if deadline_check and deadline_check() < 1.0:
            break
        if x in chosen:
            continue
            
        if _is_valid_addition(current_sums_bitset, x, n_chosen):
            chosen.append(x)
            current_sums_bitset = current_sums_bitset | (current_sums_bitset << x) | (1 << x)
            n_chosen += 1
            
    return chosen

def _local_search(initial_set, candidates, deadline_check=None, max_iters=1000):
    """
    Try to improve the set by swapping elements or adding new ones.
    """
    current_set = list(initial_set)
    current_sums = _get_sums_bitset(current_set)
    current_n = len(current_set)
    best_set = list(current_set)
    best_n = current_n
    
    # Precompute valid candidates not in set
    available = [c for c in candidates if c not in current_set]
    
    rng = random.Random(123)
    
    for _ in range(max_iters):
        if deadline_check and deadline_check() < 1.0:
            break
            
        # Try to add a random element
        if available:
            x = rng.choice(available)
            if _is_valid_addition(current_sums, x, current_n):
                current_set.append(x)
                current_sums = current_sums | (current_sums << x) | (1 << x)
                current_n += 1
                available.remove(x)
                if current_n > best_n:
                    best_n = current_n
                    best_set = list(current_set)
        
        # Try to swap an element
        if current_n > 0 and available:
            idx_to_swap = rng.randrange(current_n)
            x_to_remove = current_set[idx_to_swap]
            
            # Find new sums without x_to_remove
            # This is tricky with bitsets. We need to recompute or have a way to remove.
            # Recomputing is O(N * 2^N) which is slow.
            # Let's just try to find a replacement y such that removing x and adding y is valid.
            # We can check validity of (current_set - {x}) + {y} by building the bitset.
            
            # Optimization: Only try a few swaps per iteration to save time
            for y in rng.sample(available, min(5, len(available))):
                new_set = current_set.copy()
                new_set[idx_to_swap] = y
                if len(new_set) != len(set(new_set)):
                    continue
                
                new_sums = _get_sums_bitset(new_set)
                if new_sums.bit_count() != (1 << current_n) - 1:
                    continue
                if _check_no_divisibility_bitset(new_sums):
                    current_set = new_set
                    current_sums = new_sums
                    # Note: available list needs update
                    available.remove(y)
                    available.append(x_to_remove)
                    if current_n > best_n:
                        best_n = current_n
                        best_set = list(current_set)
                    break # Accept the swap and break inner loop
        
    return best_set

def construct():
    global _START
    _START = time.time()
    
    pool = list(range(1, 257))
    
    best = []
    best_len = 0
    
    # Strategy 1: Deterministic Greedy with different orderings
    # Order 1: Descending
    order1 = sorted(pool, reverse=True)
    result1 = _greedy_construct(order1, deadline_check=_remaining)
    if len(result1) > best_len:
        best = result1
        best_len = len(result1)
        
    # Order 2: Ascending
    if _remaining() > 5.0:
        order2 = sorted(pool)
        result2 = _greedy_construct(order2, deadline_check=_remaining)
        if len(result2) > best_len:
            best = result2
            best_len = len(result2)
            
    # Strategy 2: Randomized Restarts + Local Search
    rng = random.Random(42)
    
    while _remaining() > 2.0:
        random_pool = pool[:]
        rng.shuffle(random_pool)
        
        # Greedy start
        current_best = _greedy_construct(random_pool, deadline_check=_remaining)
        
        if len(current_best) > best_len:
            best = current_best
            best_len = len(current_best)
            
        # Local search on the best found so far (or this one if it's better)
        if _remaining() > 2.0:
            # Use the global best for local search to try to improve it
            improved = _local_search(best, pool, deadline_check=_remaining, max_iters=200)
            if len(improved) > best_len:
                best = improved
                best_len = len(best)
                
    # Final validation
    if best and not _validate_full(best):
        # Fallback to a known valid small set
        # [129, 255] is valid
        best = [129, 255]
        
    # Ensure Python ints
    return [int(x) for x in best]
