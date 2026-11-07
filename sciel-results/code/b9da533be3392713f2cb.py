import time
import random
import math

# Deadline guard: 50s wall-clock (10s safety margin under 60s limit)
_DEADLINE = 50.0
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

def _check_no_divisibility_bitset(sums_bitset, max_val=256*30):
    """
    Check if no two distinct values in the bitset divide one another.
    sums_bitset: int where bit i is set if i is a subset sum.
    """
    if sums_bitset == 0:
        return True
    
    # Extract the list of set bits (the actual sums)
    # For k up to 10, number of sums <= 1024.
    # We iterate through possible sums.
    # Optimization: We only need to check if s divides t for s < t.
    # So for each s in sums, check if 2s, 3s, ... are in sums.
    
    # Get list of sums for iteration
    # Using a loop to extract bits is fast in Python for sparse bitsets if we do it smartly,
    # but converting to list is fine for < 2^15 bits.
    sums_list = []
    temp = sums_bitset
    # We know max sum is sum(elements) <= 30*256 = 7680
    # We can iterate 1 to max_sum
    max_sum = 0
    # Find max sum quickly
    if sums_bitset:
        max_sum = sums_bitset.bit_length() - 1
    else:
        return True
        
    # Extract sums
    for i in range(1, max_sum + 1):
        if (sums_bitset >> i) & 1:
            sums_list.append(i)
            
    # Check divisibility
    # For each s in sums_list, check if any multiple 2s, 3s... is in the set.
    # We can use the bitset for O(1) lookup.
    for s in sums_list:
        # Check multiples
        multiple = s * 2
        while multiple <= max_sum:
            if (sums_bitset >> multiple) & 1:
                return False
            multiple += s
            
    return True

def _is_valid_addition_bitset(current_sums_bitset, x):
    """
    Check if adding x to a set with current_sums_bitset is valid.
    Returns True if valid.
    """
    # New sums = old sums | (old sums shifted by x) | (1 << x)
    new_sums = current_sums_bitset | (current_sums_bitset << x) | (1 << x)
    
    # Check distinctness? 
    # The problem states: "Equal sum values are counted once. Two different subsets with the same sum still make the set invalid".
    # Wait, the task says: "Two different subsets with the same sum still make the set invalid: removing their common elements leaves two disjoint subsets with the same sum s, and their union has sum 2s, which s divides."
    # This implies that if two *different* subsets have the same sum, it is INVALID.
    # My bitset approach only tracks *which* sums are achievable, not *how many* ways.
    # If two subsets have the same sum, the bit for that sum is still just 1.
    # So `_check_no_divisibility_bitset` on the resulting bitset might pass even if there are duplicate sums (which are forbidden).
    
    # We MUST check for duplicate sums (subset sum distinctness).
    # How to check if all subset sums are distinct?
    # If all subset sums are distinct, the number of non-empty subset sums must be exactly 2^n - 1.
    # So, if we have k elements, and the number of set bits in `new_sums` is not 2^k - 1, then there are collisions (duplicate sums).
    
    # We need to know k (current size).
    # Let's pass k or compute it.
    # Actually, we can just count the bits.
    
    # But wait, the prompt says: "Maximize set size."
    # If we have collisions, the set is invalid.
    # So `len(elements)` must equal `log2(popcount(new_sums) + 1)`.
    
    return None # Placeholder, logic below

def _validate_full(elements):
    """
    Full validation: 
    1. Distinct elements in 1..256.
    2. All non-empty subset sums are distinct.
    3. No two distinct subset sums divide one another.
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
    
    # Check distinctness of sums:
    # Number of non-empty subsets is 2^n - 1.
    # If popcount(sums_bitset) != 2^n - 1, there are duplicate sums.
    if sums_bitset.bit_count() != (1 << n) - 1:
        return False
        
    # Check divisibility
    if not _check_no_divisibility_bitset(sums_bitset):
        return False
        
    return True

def _greedy_construct_bitset(candidates, deadline_check=None, bias_large=True):
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
            
        # Tentatively add x
        # new_sums = current | (current << x) | (1 << x)
        new_sums = current_sums_bitset | (current_sums_bitset << x) | (1 << x)
        
        # Check distinctness:
        # Expected count is 2^(n_chosen+1) - 1
        expected_count = (1 << (n_chosen + 1)) - 1
        if new_sums.bit_count() != expected_count:
            # Collision detected, skip
            continue
            
        # Check divisibility
        # We can do a quick check. 
        # To speed up, we can check if any existing sum s divides x or x divides s?
        # Actually, just running the full check on the small bitset is fast.
        if _check_no_divisibility_bitset(new_sums):
            chosen.append(x)
            current_sums_bitset = new_sums
            n_chosen += 1
            
    return chosen

def construct():
    global _START
    _START = time.time()
    
    pool = list(range(1, 257))
    
    best = []
    best_len = 0
    
    # Pre-compute primes for bias
    primes = []
    for i in range(2, 257):
        is_prime = True
        for j in range(2, int(math.sqrt(i)) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
            
    # Strategy 1: Deterministic Greedy with different orderings
    # Order 1: Descending (Large numbers first)
    order1 = sorted(pool, reverse=True)
    result1 = _greedy_construct_bitset(order1, deadline_check=_remaining)
    if len(result1) > best_len:
        best = result1
        best_len = len(result1)
        
    # Order 2: Ascending
    if _remaining() > 5.0:
        order2 = sorted(pool)
        result2 = _greedy_construct_bitset(order2, deadline_check=_remaining)
        if len(result2) > best_len:
            best = result2
            best_len = len(result2)
            
    # Order 3: Primes first, then rest
    if _remaining() > 5.0:
        non_primes = [x for x in pool if x not in primes]
        order3 = primes + non_primes
        result3 = _greedy_construct_bitset(order3, deadline_check=_remaining)
        if len(result3) > best_len:
            best = result3
            best_len = len(result3)
            
    # Strategy 2: Randomized Restarts
    # Since the bitset check is very fast, we can do many restarts.
    rng = random.Random(42)
    
    # We will try to improve on the current best by doing local search or just random restarts.
    # Given the time budget, random restarts with greedy filling are effective.
    
    while _remaining() > 2.0:
        # Create a biased random order
        # Bias towards larger numbers and primes?
        # Let's just shuffle, but maybe sort by a random key that favors primes/large numbers?
        # Simple shuffle is a good baseline.
        random_pool = pool[:]
        rng.shuffle(random_pool)
        
        # Optional: Try to start with the best found set and add more?
        # If we have a set of size K, we can try to add elements to it.
        # But greedy from scratch is simpler and often finds good local optima.
        
        result = _greedy_construct_bitset(random_pool, deadline_check=_remaining)
        
        if len(result) > best_len:
            best = result
            best_len = len(result)
            
            # If we found a very good set, we can try to "polish" it with local search?
            # For now, let's just keep searching.
            
    # Final validation
    if best and not _validate_full(best):
        # Fallback to a known valid small set
        # [128, 192, 224] might be invalid? 
        # Let's use a very safe small set.
        # 100, 200, 250?
        # Sums: 100, 200, 250, 300, 350, 450, 550.
        # 100 divides 200, 300, 400(no), 500(no). 100 divides 200. Invalid.
        # We need a set where no sum divides another.
        # [129, 255]?
        # Sums: 129, 255, 384.
        # 129 does not divide 255 (255/129 = 1.97).
        # 129 does not divide 384 (384/129 = 2.97).
        # 255 does not divide 384.
        # Distinct? Yes.
        # So [129, 255] is valid.
        best = [129, 255]
        
    # Ensure Python ints
    return [int(x) for x in best]
