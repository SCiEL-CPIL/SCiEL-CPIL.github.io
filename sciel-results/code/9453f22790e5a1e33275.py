"""
Strategy: Randomized Greedy Construction with Integer-Packed Vector Arithmetic.

Instead of restricting the search to prime powers (which the current program does),
we search the entire space of integers 2..40.
The key bottleneck in checking distinct subset products is the enumeration and comparison
of $2^k$ products. 
For $k \approx 18$, $2^{18} \approx 2.6 \times 10^5$. 
Comparing big integers or dictionaries is slow.
We represent each number by its prime exponent vector.
Since the numbers are $\le 40$, the primes involved are $\{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37\}$.
There are 12 primes.
The maximum exponent for 2 in a number $\le 40$ is 5 ($32=2^5$).
For 3, max is 3 ($27=3^3$).
For 5, max is 2 ($25=5^2$).
For others, max is 1.
We can pack these exponents into a single Python integer.
Let the primes be indexed $p_0=2, p_1=3, \dots, p_{11}=37$.
We need enough bits for each exponent.
Max exponent for 2 is 5 (needs 3 bits).
Max exponent for 3 is 3 (needs 2 bits).
Max exponent for 5 is 2 (needs 2 bits).
Max exponent for others is 1 (needs 1 bit).
Total bits: $3 + 2 + 2 + 9 \times 1 = 16$ bits.
This fits easily in a 64-bit integer (or Python int).
The subset product corresponds to the sum of the packed vectors.
Since the packing is linear (sum of exponents corresponds to sum of packed ints, provided no carry-over occurs between fields),
we can check distinctness by generating all $2^k$ sums and putting them in a set.
If the set size is $2^k$, they are distinct.

This check is $O(2^k)$ but with very fast integer addition and hashing.
We use a randomized greedy approach:
1. Start with an empty set.
2. Repeatedly pick a random candidate from the remaining numbers.
3. If adding it maintains distinctness, add it.
4. If stuck, try local search (swap) or restart.
5. Use multiple random restarts to explore different local maxima.
"""

import time
import random
from functools import lru_cache

# Primes in 2..40
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
NUM_PRIMES = len(PRIMES)

# Bit widths for each prime's exponent in the packed representation
# We need to ensure no carry-over.
# Max exponent for 2 in [2..40] is 5 (2^5=32). 5 < 8. Let's use 4 bits to be safe.
# Max exponent for 3 is 3 (3^3=27). 3 < 8. 4 bits.
# Max exponent for 5 is 2. 4 bits.
# For primes >= 7, max exponent is 1. 4 bits is overkill but simplifies code.
# 12 primes * 4 bits = 48 bits. Fits in 64-bit int.
BITS_PER_EXP = 4
MASK = (1 << BITS_PER_EXP) - 1

# Precompute packed vector for each number 2..40
PACKED_VECTORS = {}
def get_packed_vector(n):
    if n in PACKED_VECTORS:
        return PACKED_VECTORS[n]
    vec = 0
    for i, p in enumerate(PRIMES):
        if n % p == 0:
            exp = 0
            while n % p == 0:
                n //= p
                exp += 1
            vec |= (exp << (i * BITS_PER_EXP))
    PACKED_VECTORS[n] = vec
    return vec

# Precompute for all candidates
ALL_NUMS = list(range(2, 41))
for n in ALL_NUMS:
    get_packed_vector(n)

def check_distinct_subset_products_packed(numbers):
    """
    Check if all subset products are distinct using packed integer vectors.
    Returns True if distinct.
    """
    k = len(numbers)
    if k == 0:
        return True
    
    vecs = [get_packed_vector(n) for n in numbers]
    
    # Generate all 2^k subset sums
    # We can do this iteratively or using bit masks.
    # For k up to 18-20, 2^k is manageable.
    # Optimization: If we maintain a set of current subset sums, adding a new vector v
    # generates new sums {s + v for s in current_sums}.
    # We check if any new sum is in current_sums.
    # This is O(2^k) per check, but we can do it incrementally if we build the set step-by-step.
    # However, for a candidate check, we just need to verify the whole set.
    
    # Fast path: use a set of sums
    sums = {0}
    for v in vecs:
        new_sums = set()
        for s in sums:
            new_s = s + v
            if new_s in sums or new_s in new_sums:
                # Collision detected
                return False
            new_sums.add(new_s)
        sums.update(new_sums)
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_set = []
    best_size = 0
    
    # Strategy: Randomized Greedy with Restarts
    # We try to build a set by greedily adding numbers.
    # To avoid local minima, we use random ordering and multiple restarts.
    # Also, we try to "upgrade" the set by swapping elements if it allows adding more.
    
    # Pre-shuffled list of all numbers
    all_nums = ALL_NUMS[:]
    
    # Number of restarts depends on time available
    # Each check is O(2^k * k) roughly. For k=18, 2^18 ~ 260k.
    # 260k operations is fast in Python if using sets of ints.
    # Let's estimate: 260k additions and set lookups ~ 0.05s.
    # We can do ~100-200 restarts in 85s?
    # Actually, the greedy step itself calls check_distinct_subset_products_packed.
    # If we add one by one, the cost is sum_{i=1}^k 2^i = 2^{k+1}.
    # So building a set of size 18 costs ~ 2^19 = 524,288 operations.
    # This is very fast.
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        
        # Randomize the order of candidates
        random.shuffle(all_nums)
        
        current_set = []
        # Greedy add
        for n in all_nums:
            # Check if we can add n
            # Optimization: Only check if we are close to the deadline or if the set is small?
            # No, just check.
            if time.time() > deadline - 1.0:
                break
            
            # Try adding n
            # To speed up, we can check if n is "compatible" with the current set.
            # A quick necessary condition: n must not be a product of a subset of current_set?
            # Actually, the check_distinct_subset_products_packed does the full check.
            # We can optimize by maintaining the set of subset sums for current_set.
            # But for simplicity and robustness, let's just call the check function.
            # Wait, calling the full check for every candidate is expensive if k is large.
            # Let's maintain the set of sums incrementally.
            
            # Let's refactor to maintain sums.
            # This is a local optimization for the greedy step.
            
            # For now, use the full check. It's O(2^k).
            # If k=15, 2^15=32768. Very fast.
            # If k=18, 2^18=262144. Still fast.
            
            # Let's try adding n
            test_set = current_set + [n]
            if check_distinct_subset_products_packed(test_set):
                current_set = test_set
        
        # Local search: Try to improve by swapping
        # If we are stuck, maybe swapping an element allows adding a new one.
        # We do a few rounds of swaps.
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try swapping each element in current_set with a candidate not in it
            # This is O(k * (40-k)) checks.
            # Each check is O(2^k).
            # This might be too slow if k is large.
            # Let's limit the number of swap attempts.
            
            # Shuffle candidates for randomization
            candidates = [n for n in all_nums if n not in current_set]
            random.shuffle(candidates)
            
            # Try a limited number of swaps
            swap_attempts = 0
            max_swap_attempts = 50 # Limit per restart
            
            for i in range(len(current_set)):
                for n in candidates:
                    if swap_attempts >= max_swap_attempts:
                        break
                    swap_attempts += 1
                    
                    # Swap current_set[i] with n
                    test_set = current_set[:i] + [n] + current_set[i+1:]
                    if check_distinct_subset_products_packed(test_set):
                        # Check if this new set allows adding another number
                        # We can try to greedily add to this new set
                        new_set = test_set[:]
                        # Try adding any remaining candidate
                        for m in candidates:
                            if m == n:
                                continue
                            if m in new_set:
                                continue
                            test_add = new_set + [m]
                            if check_distinct_subset_products_packed(test_add):
                                new_set = test_add
                                improved = True
                                break
                        if len(new_set) > len(current_set):
                            current_set = new_set
                            # Update candidates
                            candidates = [n for n in all_nums if n not in current_set]
                            random.shuffle(candidates)
                            break
                        elif len(new_set) == len(current_set):
                            # Even if size is same, maybe this configuration is better?
                            # For now, we only accept if size increases.
                            # If we accept same size, we might loop.
                            # Let's only accept if size increases.
                            pass
                    if improved:
                        break
                if improved:
                    break
            
            # If we didn't improve size, we might be stuck.
            # We could try a "kick" move: remove one element and try to add two.
            # But that's complex. Random restarts should handle it.
            
        # Update best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            # If we reached 18, we can stop early?
            # The max possible is 18 (from 2..40, 39 numbers).
            # But 18 is a very high bar.
            # If we find 18, it's likely optimal or near-optimal.
            if best_size >= 18:
                break

    # Final verification
    if not check_distinct_subset_products_packed(best_set):
        # Fallback to a known good set (prime powers)
        # We need to generate a valid prime power set.
        # From the previous program, {2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37} was valid (size 16).
        best_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    # Ensure types are Python int
    return [int(x) for x in best_set]

# To make it more robust, let's add a specific heuristic:
# Prime powers are "safe". Composites are "risky".
# We can bias the random shuffle to prefer prime powers initially, then explore composites.
# Or, we can just rely on the random restarts.

# Let's refine the greedy step to be faster by maintaining the sum set.
# This avoids re-computing all sums from scratch for each candidate.

def construct_fast():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    best_size = 0
    
    all_nums = ALL_NUMS[:]
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        
        random.shuffle(all_nums)
        
        current_set = []
        current_sums = {0}
        
        for n in all_nums:
            if time.time() > deadline - 1.0:
                break
                
            v = get_packed_vector(n)
            
            # Check if adding v creates a collision
            # New sums will be {s + v for s in current_sums}
            # Collision if any s + v is in current_sums
            # Note: s + v == s' + v implies s == s', so no internal collision in new_sums.
            # We only need to check intersection with current_sums.
            
            collision = False
            for s in current_sums:
                if s + v in current_sums:
                    collision = True
                    break
            
            if not collision:
                # Safe to add
                current_set.append(n)
                # Update sums
                new_sums = set()
                for s in current_sums:
                    new_sums.add(s + v)
                current_sums.update(new_sums)
        
        # Local search with maintained sums?
        # Swapping is harder with maintained sums because removing an element requires
        # subtracting its vector from all sums, which is not straightforward with sets.
        # We can revert to full check for swaps.
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            candidates = [n for n in all_nums if n not in current_set]
            random.shuffle(candidates)
            
            # Try limited swaps
            for i in range(len(current_set)):
                for n in candidates[:20]: # Limit candidates per swap position
                    # Swap current_set[i] with n
                    test_set = current_set[:i] + [n] + current_set[i+1:]
                    if check_distinct_subset_products_packed(test_set):
                        # Try to add more
                        new_set = test_set[:]
                        # Rebuild sums for new_set? Or just use full check for additions?
                        # Let's use full check for simplicity in this inner loop.
                        for m in candidates:
                            if m == n or m in new_set:
                                continue
                            test_add = new_set + [m]
                            if check_distinct_subset_products_packed(test_add):
                                new_set = test_add
                                improved = True
                                break
                        if len(new_set) > len(current_set):
                            current_set = new_set
                            candidates = [n for n in all_nums if n not in current_set]
                            random.shuffle(candidates)
                            break
                if improved:
                    break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            if best_size >= 18:
                break

    if not check_distinct_subset_products_packed(best_set):
        best_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    return [int(x) for x in best_set]

# Use the fast version
def construct():
    return construct_fast()
