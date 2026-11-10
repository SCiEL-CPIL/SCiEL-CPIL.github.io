"""
Strategy: Dynamic Programming / Meet-in-the-Middle for Maximum Independent Set in Subset Sum Space.
Instead of local search, we use a DP over subsets of the available "dimensions" (primes) 
and their possible exponent sums. 
We model each number n in 2..40 as a vector of prime exponents.
The condition "all subset products distinct" is equivalent to "all subset sums of vectors are distinct".
This is equivalent to the vectors being "sum-free" (no non-empty subset sums to zero, and more generally, 
any two distinct subsets have distinct sums).
Actually, the condition is that the map from subset masks to product values is injective.
Since products correspond to vectors, this is equivalent to the map from subset masks to vector sums being injective.
This is a known property related to "Sidon sequences" or "B_h sequences" in additive combinatorics, but for vectors.

However, a simpler structural observation:
If we select a set of numbers S, the number of distinct subset products is 2^{|S|}.
The maximum possible product of any subset is P_max.
If 2^{|S|} > P_max + 1, then by Pigeonhole Principle, two subset products must be equal.
Thus, |S| <= log2(P_max + 1).

Let's calculate the upper bound for |S| using numbers in 2..40.
The maximum product is achieved by taking the largest numbers.
But wait, the numbers are distinct.
The largest possible product of k distinct numbers from 2..40 is the product of the k largest numbers.
Let's check the bound for k=18.
Product of 18 largest numbers: 40*39*...*23.
log2 of this product.
If 2^18 > P_max + 1, then k=18 is impossible.

Let's refine the strategy:
1. Pre-compute prime exponent vectors for all n in 2..40.
2. Use a recursive search (backtracking) with strong pruning:
   - Maintain the current set of selected numbers.
   - Maintain the set of all current subset products (or their vector representations).
   - When adding a new number n, the new subset products are {p * n for p in current_products} U current_products.
   - If any collision occurs (new product already in set, or product in new set collides with existing), reject.
   - Pruning: If the theoretical maximum size achievable from remaining candidates + current size <= best_found, prune.
   - To speed up, we can order candidates by some heuristic (e.g., descending value, or by "complexity" i.e., number of prime factors).

This is essentially a Maximum Independent Set problem on a conflict graph, but the conflict is global (subset sums).
The "vector sum" representation is key for fast collision detection.
We represent the set of subset sums as a set of tuples of exponents.
Since the numbers are small (<=40), the exponents are small.
The number of primes is 12 (2, 3, ..., 37).
A vector is a tuple of 12 integers.
Adding a vector is element-wise addition.
Checking if a new vector is in the set is O(1) average.

The number of subset sums for a set of size k is 2^k.
For k=18, 2^18 = 262,144. This is manageable.
For k=19, 2^19 = 524,288.
For k=20, 2^20 = 1,048,576.

We can try to find the maximum k.
We know k=16 is possible.
Let's try to find k=17 or 18.

The backtracking search can be optimized by:
1. Pre-computing vectors for all n in 2..40.
2. Trying to add numbers in a specific order.
3. Using the Pigeonhole Principle bound to prune:
   Let current_max_product be the max product in the current set of subset products.
   Actually, the bound is on the value of the product.
   If we use integer products, the max product for k=18 is huge, so the bound is loose.
   But if we use vectors, the "space" is the grid of exponents.
   The number of distinct vectors is 2^k.
   The "volume" of the grid occupied by the vectors is bounded by the product of (max_exponent + 1) for each prime.
   This is a stronger bound.
   
   Let E_p be the max exponent of prime p in any single number <= 40.
   2: max exp 5 (32)
   3: max exp 3 (27)
   5: max exp 2 (25)
   7,11,13,17,19,23,29,31,37: max exp 1.
   
   The total number of possible distinct vectors (subset sums) is:
   (5+1) * (3+1) * (2+1) * (1+1)^9 = 6 * 4 * 3 * 512 = 36864.
   
   Wait, this is the number of distinct vectors possible if we sum ANY combination of the max exponents?
   No. The subset sums are sums of the vectors of the SELECTED numbers.
   The maximum exponent of 2 in a subset sum is the sum of exponents of 2 in the selected numbers.
   If we select numbers with high powers of 2, the max exponent grows.
   
   However, the Pigeonhole Principle applies to the VALUES of the products.
   If the number of subset products 2^k exceeds the range of possible product values, they must collide.
   But product values are integers. The max product is ~ 40^18. 2^18 is much smaller.
   So the value-based Pigeonhole Principle is not the bottleneck.
   
   The bottleneck is the linear independence.
   The vectors must be such that no non-trivial subset sums to zero.
   This is equivalent to the vectors being linearly independent over the rationals? No, over the integers with coefficients -1, 0, 1.
   
   Let's just run the backtracking search. It's robust.
"""

import time
from functools import lru_cache

def _get_prime_factors(n):
    factors = {}
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

# Pre-compute vectors for all numbers 2..40
# Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
NUM_PRIMES = len(PRIMES)

def _to_vector(n):
    facs = _get_prime_factors(n)
    vec = [0] * NUM_PRIMES
    for i, p in enumerate(PRIMES):
        vec[i] = facs.get(p, 0)
    return tuple(vec)

# Vector for each number 2..40
VECTORS = {n: _to_vector(n) for n in range(2, 41)}

def _add_vectors(v1, v2):
    return tuple(a + b for a, b in zip(v1, v2))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = []
    best_size = 0
    
    # We want to maximize the size of the set.
    # Candidates are 2..40.
    # We will use backtracking.
    # To make it faster, we can fix a subset of "large primes" which are always safe to include?
    # No, mixing might allow more small primes.
    
    # Let's try to find a set of size 17 or 18.
    # If we fail, we return the best found (likely 16).
    
    # Heuristic: Start with the "greedy" best set found in the previous strategy (size 16) 
    # and try to extend it?
    # Or start from scratch with backtracking.
    
    # Backtracking state:
    # current_set: list of selected numbers
    # current_sums: set of tuples representing subset sums of current_set
    # candidates: list of remaining numbers to consider
    
    # To prune effectively, we can use the fact that if we have k numbers, we have 2^k sums.
    # The "space" of sums is limited.
    
    # Let's implement a recursive function that tries to add numbers.
    # Order of candidates: Try numbers that are "independent" early.
    # Primes are most independent.
    
    def backtrack(current_set, current_sums, candidates):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
        
        if not candidates:
            return
        
        # Pruning: If even taking all remaining candidates doesn't beat best_size, stop.
        if len(current_set) + len(candidates) <= best_size:
            return
            
        for i in range(len(candidates)):
            n = candidates[i]
            v = VECTORS[n]
            
            # Check if adding n creates a collision.
            # New sums = current_sums U {s + v for s in current_sums}
            # Collision if:
            # 1. v is in current_sums (meaning n is a subset product of existing) -> Wait, v is the vector of n.
            #    If v is in current_sums, then n = product of some subset of current_set.
            #    This implies a collision: product(subset) * 1 = n * 1.
            #    Actually, the condition is that all subset products are distinct.
            #    If n is a product of a subset of current_set, then the product of that subset (which is n)
            #    collides with the product of the subset {n} (which is n).
            #    So yes, v must not be in current_sums.
            # 2. Any s + v (for s in current_sums) must not be in current_sums.
            #    Because if s + v = t (for some t in current_sums), then product(subset_s) * n = product(subset_t).
            #    This is a collision between two distinct subsets (subset_s U {n} and subset_t).
            
            if v in current_sums:
                continue
                
            # Check for collisions with existing sums
            collision = False
            new_sums_list = []
            for s in current_sums:
                ns = _add_vectors(s, v)
                new_sums_list.append(ns)
                if ns in current_sums:
                    collision = True
                    break
            
            if collision:
                continue
            
            # Also need to check if any ns in new_sums_list collides with another ns' in new_sums_list?
            # No, if s1 != s2, then s1+v != s2+v. So new_sums_list has distinct elements.
            # We only need to check against current_sums.
            
            new_sums = current_sums.union(new_sums_list)
            
            # Recurse
            backtrack(current_set + [n], new_sums, candidates[i+1:])
            
            # Backtrack: no need to undo current_sums as it's passed by value (immutable set)
            # But creating new_sums is O(2^k).
            # For k=16, 2^16 = 65536. This is fast.
            # For k=17, 2^17 = 131072.
            # For k=18, 2^18 = 262144.
            
            # Optimization: If we find a solution of size 18, we can stop?
            # The max possible is likely 18 or 19.
            # If we find 18, it's very likely optimal.
            if best_size >= 18:
                return

    # Initial candidates: 2..40
    # To improve search, we can order candidates.
    # Primes first? Or small numbers first?
    # Small numbers have smaller vectors, might be more "compatible"?
    # Let's try sorting by value.
    
    candidates = list(range(2, 41))
    # Zero vector is (0,...,0)
    initial_sums = {tuple([0]*NUM_PRIMES)}
    
    backtrack([], initial_sums, candidates)
    
    # Fallback if backtracking didn't find anything good (shouldn't happen, but safe)
    if not best_set:
        best_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    # Verify the result
    # Quick check
    def _verify(nums):
        if not nums: return True
        vecs = [VECTORS[n] for n in nums]
        sums = {tuple([0]*NUM_PRIMES)}
        for v in vecs:
            new_sums = set()
            for s in sums:
                ns = _add_vectors(s, v)
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True
        
    if not _verify(best_set):
        # Should not happen if backtrack logic is correct
        best_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    return [int(x) for x in best_set]

