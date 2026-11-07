"""Program for cardinality-separated subset sums using bitset DP verification."""
import time
import random
from itertools import combinations

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ. Maximize set size.
    
    Uses greedy construction with bitset-based cardinality-indexed subset-sum DP.
    """
    deadline = time.time() + 85  # safety margin before 90s kill
    
    N = 256  # max value
    
    def is_valid_cardinality_separated(elements):
        """
        Check if the set satisfies: for all k1 != k2, no subset of size k1 
        has the same sum as a subset of size k2.
        
        We track for each cardinality k, the set of achievable sums.
        Then check that these sets are pairwise disjoint across different k.
        """
        n = len(elements)
        if n == 0:
            return True
        
        max_sum = sum(elements)
        
        # sums_by_card[k] = bitset where bit s is set iff sum s is achievable with k elements
        # Use list of Python ints as bitsets
        # sums_by_card[0] = 1 (sum 0 with 0 elements)
        sums_by_card = [0] * (n + 1)
        sums_by_card[0] = 1  # bit 0 set
        
        for x in elements:
            # Update in reverse to avoid using x multiple times
            for k in range(n, 0, -1):
                # Add x to all sums achievable with k-1 elements
                sums_by_card[k] |= (sums_by_card[k-1] << x)
        
        # Check pairwise disjointness across different cardinalities
        # For each pair (k1, k2) with k1 != k2, check (sums_by_card[k1] & sums_by_card[k2]) == 0
        for k1 in range(n + 1):
            for k2 in range(k1 + 1, n + 1):
                if sums_by_card[k1] & sums_by_card[k2]:
                    return False
        return True
    
    def try_greedy(start_elements, available_pool, order_func):
        """
        Try to grow the set greedily by adding elements from available_pool
        in the order given by order_func, checking validity each time.
        """
        current = list(start_elements)
        current_set = set(current)
        remaining = [x for x in available_pool if x not in current_set]
        remaining.sort(key=order_func)
        
        for x in remaining:
            if time.time() > deadline:
                break
            candidate = current + [x]
            if is_valid_cardinality_separated(candidate):
                current = candidate
                current_set.add(x)
        
        return current
    
    # Generate diverse starting bases
    # Strategy: start with small sets that are likely valid, then grow
    
    # Base 1: consecutive integers starting from various points
    # For a set {a, a+1, ..., a+n-1}, the min k-sum is k*a + k(k-1)/2
    # The max (k-1)-sum is (k-1)*(a+n-1) + (k-1)(k-2)/2
    # We need min k-sum > max (k-1)-sum for all k >= 2
    # k*a + k(k-1)/2 > (k-1)*(a+n-1) + (k-1)(k-1)/2
    # k*a + k(k-1)/2 > (k-1)*a + (k-1)(n-1) + (k-1)(k-1)/2
    # a + k(k-1)/2 > (k-1)(n-1) + (k-1)(k-1)/2
    # a > (k-1)(n-1) + (k-1)(k-1)/2 - k(k-1)/2
    # a > (k-1)(n-1) - (k-1)/2
    # a > (k-1)(n - 3/2)
    
    # For n=60, k=60: a > 59*(60-1.5) = 59*58.5 = 3451.5 -- way too large
    # So consecutive integers won't work for large sets.
    
    # Better approach: use powers of 2 or superincreasing sequences
    # For superincreasing: each element > sum of all previous
    # Then subset sums are unique, so certainly cardinality-separated
    
    # But we want to maximize size within 1..256.
    # Superincreasing: 1, 2, 4, 8, 16, 32, 64, 128 -> only 8 elements
    
    # We need a smarter approach. Let's use the bitset DP to verify.
    
    # Strategy: Try multiple random/greedy approaches
    
    best_set = []
    
    # Approach 1: Start with small base, grow greedily
    # Try different base sizes and values
    
    # First, let's try a simple greedy from small to large
    # Start with {1}, try to add 2, 3, 4, ...
    
    # Actually, let's think about what makes a set valid.
    # The condition is that subset sums of different cardinalities don't overlap.
    # This is a strong condition.
    
    # Let's try: start with a base, then greedily add the smallest available element that maintains validity.
    
    # Multiple strategies:
    strategies = []
    
    # Strategy A: Start empty, add elements in increasing order
    # Strategy B: Start with a few large elements, add smaller ones
    # Strategy C: Random restarts
    
    # Let's implement a general greedy that tries to add elements in a given order
    
    def greedy_grow(base, pool, order):
        """Grow base by adding elements from pool in given order if valid."""
        current = list(base)
        current_set = set(current)
        
        # Sort pool by order
        sorted_pool = sorted([x for x in pool if x not in current_set], key=order)
        
        for x in sorted_pool:
            if time.time() > deadline:
                break
            candidate = current + [x]
            if is_valid_cardinality_separated(candidate):
                current = candidate
                current_set.add(x)
        
        return current
    
    # Try increasing order
    pool = list(range(1, 257))
    
    # Start with empty set
    result = greedy_grow([], pool, lambda x: x)
    if len(result) > len(best_set):
        best_set = result
    
    # Start with {1}
    result = greedy_grow([1], pool, lambda x: x)
    if len(result) > len(best_set):
        best_set = result
    
    # Start with {1, 2}
    result = greedy_grow([1, 2], pool, lambda x: x)
    if len(result) > len(best_set):
        best_set = result
    
    # Try descending order (add largest first)
    result = greedy_grow([], pool, lambda x: -x)
    if len(result) > len(best_set):
        best_set = result
    
    # Try even-odd order
    evens = [x for x in pool if x % 2 == 0]
    odds = [x for x in pool if x % 2 == 1]
    eo_order = {x: (0, i) for i, x in enumerate(evens)}
    eo_order.update({x: (1, i) for i, x in enumerate(odds)})
    result = greedy_grow([], pool, lambda x: eo_order[x])
    if len(result) > len(best_set):
        best_set = result
    
    # Try alternating large-small
    # Sort descending, take every other
    sorted_desc = sorted(pool, reverse=True)
    alt_order = {}
    for i, x in enumerate(sorted_desc):
        if i % 2 == 0:
            alt_order[x] = (0, i//2)
        else:
            alt_order[x] = (1, i//2)
    result = greedy_grow([], pool, lambda x: alt_order[x])
    if len(result) > len(best_set):
        best_set = result
    
    # Random restarts
    rng = random.Random(42)
    for _ in range(50):
        if time.time() > deadline:
            break
        # Random base of size 1-3
        base_size = rng.randint(1, 3)
        base = rng.sample(pool, base_size)
        # Random order
        random_pool = pool[:]
        rng.shuffle(random_pool)
        order_dict = {x: i for i, x in enumerate(random_pool)}
        result = greedy_grow(base, pool, lambda x: order_dict[x])
        if len(result) > len(best_set):
            best_set = result
    
    # Try more systematic approaches
    # Start with different bases
    for start in [1, 2, 3, 4, 5, 8, 10, 16, 20, 32, 50, 100, 128, 200, 256]:
        if time.time() > deadline:
            break
        result = greedy_grow([start], pool, lambda x: x)
        if len(result) > len(best_set):
            best_set = result
    
    # Try pairs as base
    for a in [1, 2, 3, 5, 8, 10, 16, 20, 32, 50, 100, 128, 200]:
        if time.time() > deadline:
            break
        for b in [a+1, a+2, a+5, a+10, a+20, a+50, min(a*2, 256)]:
            if b > 256 or b <= a:
                continue
            result = greedy_grow([a, b], pool, lambda x: x)
            if len(result) > len(best_set):
                best_set = result
    
    # Final polish: try to add any remaining elements in any order
    # Do a final pass trying all elements in multiple orders
    if len(best_set) > 0:
        current_set = set(best_set)
        remaining = [x for x in pool if x not in current_set]
        
        # Try adding each remaining element in increasing order
        for x in sorted(remaining):
            if time.time() > deadline:
                break
            candidate = best_set + [x]
            if is_valid_cardinality_separated(candidate):
                best_set = candidate
                current_set.add(x)
        
        # Try again in decreasing order
        current_set = set(best_set)
        remaining = [x for x in pool if x not in current_set]
        for x in sorted(remaining, reverse=True):
            if time.time() > deadline:
                break
            candidate = best_set + [x]
            if is_valid_cardinality_separated(candidate):
                best_set = candidate
                current_set.add(x)
    
    # Ensure we return a list of Python ints
    return list(best_set)
