"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from itertools import combinations, product

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 55.0  # Leave margin for 90s limit

def construct():
    # 1. Find all primes <= 256
    primes = []
    for i in range(2, 257):
        is_prime = True
        for j in range(2, isqrt(i) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    
    # 2. Calculate max power for each prime <= 256
    max_powers = {}
    for p in primes:
        v = 1
        while v * p <= 256:
            v *= p
        max_powers[p] = v
        
    # 3. Identify primes that can potentially be merged
    # A prime p can be merged with q if p*q <= 256 (at least with power 1)
    # But we should check if the merged value > sum of individual max powers.
    
    # Base case: All primes separate
    base_numbers = [max_powers[p] for p in primes]
    base_sum = sum(base_numbers)
    
    # We only need to consider merging primes p where p <= 16 (since p*q <= 256 implies q >= 2, p <= 128)
    # Actually, if p > 128, it can't be merged with anything.
    # If p > 85, it can only be merged with 2.
    # Let's focus on primes <= 113 for potential merging.
    
    # To be safe and fast, we will try to find a set of disjoint pairs/triples
    # that increase the sum.
    
    # Since the number of small primes is small, we can just check all pairs
    # and use a greedy matching or simple DP.
    
    # Let's define a function to get the best value for a set of primes
    def best_value_for_set(prime_set):
        # prime_set is a frozenset or tuple of primes
        # We want max prod(p^e) <= 256
        # Since size is small (<= 3 or 4), we can iterate exponents
        if not prime_set:
            return 0
        if len(prime_set) == 1:
            return max_powers[list(prime_set)[0]]
        
        # For small sets, use recursion or product
        # Max exponent for 2 is 7, 3 is 5, etc.
        # We can just iterate possible products?
        # No, iterate exponents.
        
        # Sort primes
        ps = sorted(prime_set)
        
        def rec(idx, current_val):
            if idx == len(ps):
                return current_val
            p = ps[idx]
            max_val = 0
            v = 1
            while v * p <= 256:
                if current_val * v > 256:
                    break
                # We must pick at least power 1
                res = rec(idx + 1, current_val * v)
                if res > max_val:
                    max_val = res
                v *= p
            return max_val
            
        return rec(0, 1)

    # We will use a more robust matching approach for positive gain pairs
    # Since the number of such pairs is small, we can use a simple backtracking/DP
    # or even just iterate if the graph is small enough.
    
    small_primes = [p for p in primes if p <= 128]
    
    # List of (gain, p1_idx, p2_idx, val_together)
    # We map primes to indices for easier handling
    prime_to_idx = {p: i for i, p in enumerate(small_primes)}
    
    pairs = []
    for i in range(len(small_primes)):
        for j in range(i+1, len(small_primes)):
            p1 = small_primes[i]
            p2 = small_primes[j]
            if p1 * p2 > 256:
                continue
            val_sep = max_powers[p1] + max_powers[p2]
            val_together = best_value_for_set((p1, p2))
            gain = val_together - val_sep
            if gain > 0:
                pairs.append((gain, i, j, val_together))
                
    # Find max weight matching
    # n is number of small primes
    n = len(small_primes)
    # Use bitmask DP if n is small enough (n <= 20)
    # small_primes count is approx 30. 2^30 is too big.
    # But the graph of positive gains is very sparse.
    # We can just use recursive backtracking on the pairs.
    
    # Sort pairs by gain descending to help pruning
    pairs.sort(key=lambda x: x[0], reverse=True)
    
    best_gain = 0
    best_merges = []
    
    def backtrack(idx, used_mask, current_gain, current_merges):
        nonlocal best_gain, best_merges
        if current_gain > best_gain:
            best_gain = current_gain
            best_merges = current_merges[:]
            
        if idx == len(pairs):
            return
            
        # Option 1: Skip this pair
        backtrack(idx + 1, used_mask, current_gain, current_merges)
        
        # Option 2: Take this pair if not used
        gain, i, j, val = pairs[idx]
        if not (used_mask & (1 << i)) and not (used_mask & (1 << j)):
            backtrack(idx + 1, used_mask | (1 << i) | (1 << j), current_gain + gain, current_merges + [(small_primes[i], small_primes[j], val)])

    backtrack(0, 0, 0, [])
    
    used_primes = set()
    merges = []
    for p1, p2, val in best_merges:
        merges.append((p1, p2, val))
        used_primes.add(p1)
        used_primes.add(p2)
            
    # Construct result
    result = []
    merged_primes_set = used_primes
    
    # Add merged values
    for p1, p2, val in merges:
        result.append(val)
        
    # Add unmerged primes with max power
    for p in primes:
        if p not in merged_primes_set:
            result.append(max_powers[p])
            
    # Check for 1? 
    # The problem says "Integer one is allowed once".
    # 1 is coprime to everything.
    # Does adding 1 increase the sum? Yes, by 1.
    # But is 1 distinct from others? Yes, others are >= 2.
    # So we should always include 1.
    result.append(1)
    
    # Ensure distinctness (should be guaranteed by construction)
    # Ensure all are ints
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
