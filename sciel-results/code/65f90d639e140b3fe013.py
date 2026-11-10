"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from collections import defaultdict

P = {'n': 256, 'maximum_size': 60}

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums must differ. Repeated sums within the
    SAME cardinality are allowed. Maximize set size.
    
    Uses a global conflict-driven removal strategy:
    1. Start with a large random candidate set.
    2. Compute all subset sums by cardinality using DP.
    3. Identify cross-cardinality sum collisions (same sum appearing in different cardinalities).
    4. Remove elements involved in the most frequent collisions.
    5. Repeat until valid or size limit reached.
    """
    n_max = 256
    target_max_size = 60
    
    # We want to find a set of size as close to 60 as possible.
    # The constraint is: for any k1 != k2, the set of sums of subsets of size k1 
    # must be disjoint from the set of sums of subsets of size k2.
    
    # Let's try to build a set of size 60. If that fails, we can try smaller sizes,
    # but the task says "Maximize set size", and the seed returns 8.
    # Let's aim for 60 first.
    
    # Strategy:
    # A sufficient condition for the property to hold is that the minimum sum of a subset
    # of size k+1 is strictly greater than the maximum sum of a subset of size k.
    # This is a very strong condition. Let's see if we can satisfy it or something weaker.
    # Actually, the condition is just that the *sets* of sums are disjoint.
    
    # Let's use the iterative removal approach.
    # Start with a set of size, say, 60.
    # We need to ensure that for all k1 != k2, Sum_k1 ∩ Sum_k2 = ∅.
    
    # To check validity and find conflicts, we can compute the set of sums for each cardinality.
    # For a set S of size m, the number of subsets is 2^m, which is huge for m=60.
    # So we cannot enumerate all subsets.
    
    # Wait, the task says "Maximize set size using exact cardinality-indexed subset-sum dynamic programming."
    # This implies we should use DP to compute the *possible sums* for each cardinality.
    # The state for DP can be (cardinality, sum).
    # The maximum sum for 60 elements each <= 256 is 60*256 = 15360.
    # The DP table would be of size (61 x 15361), which is manageable.
    
    # Let's define a function to check if a given set S satisfies the condition.
    def is_valid(S):
        m = len(S)
        if m == 0:
            return True
        
        # DP: possible_sums[k] is a set of sums achievable by subsets of size k.
        # We can use a list of sets or a 2D boolean array.
        # Let's use a list of sets for clarity, or a 2D array for speed.
        # max_sum = sum(S)
        # dp[k][s] = True if there is a subset of size k with sum s.
        
        max_sum = sum(S)
        # dp[k] is a set of sums
        dp = [set() for _ in range(m + 1)]
        dp[0].add(0)
        
        for x in S:
            # Iterate backwards to avoid using x multiple times in the same step
            # But since we are building up cardinality, we need to be careful.
            # Standard 0/1 knapsack DP:
            # new_dp[k] = dp[k] ∪ {s + x | s in dp[k-1]}
            # To do this in place, we iterate k from m down to 1.
            for k in range(m, 0, -1):
                new_sums = set()
                for s in dp[k-1]:
                    new_sums.add(s + x)
                dp[k].update(new_sums)
        
        # Check for collisions
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                if dp[i] & dp[j]:
                    return False
        return True

    # This DP approach is O(m * max_sum) per element, so O(m^2 * max_sum) total?
    # Actually, the inner loop iterates over the size of dp[k-1], which can be up to max_sum.
    # So it's roughly O(m * max_sum) for one element, and O(m^2 * max_sum) for the whole set.
    # m=60, max_sum=15360. 60*60*15360 ≈ 55 million operations. This might be too slow in Python
    # if done repeatedly in a search loop.
    
    # Let's optimize the validity check.
    # We can use bitsets or numpy arrays for the DP to speed it up.
    
    def is_valid_fast(S):
        m = len(S)
        if m == 0:
            return True
        
        max_sum = sum(S)
        # dp[k] is a bitset (integer) where bit s is 1 if sum s is achievable with cardinality k.
        # Python integers are arbitrary precision, so they work as bitsets.
        dp = [0] * (m + 1)
        dp[0] = 1  # bit 0 is set
        
        for x in S:
            for k in range(m, 0, -1):
                # Shift dp[k-1] by x and OR into dp[k]
                dp[k] |= (dp[k-1] << x)
        
        # Check for collisions
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                if dp[i] & dp[j]:
                    return False
        return True

    # This is much faster. Shifting and OR-ing large integers is very fast in Python.
    
    # Now, the search strategy:
    # 1. Start with a random set of size 60 from 1..256.
    # 2. Check if it's valid.
    # 3. If not, identify which elements are involved in collisions.
    #    How to identify? We can remove one element at a time and see if the number of collisions decreases.
    #    Or, we can compute the "conflict score" for each element.
    #    A simpler heuristic: try removing each element one by one, and keep the one that results in
    #    the fewest collisions (or a valid set).
    
    # Let's try a different approach: Greedy construction.
    # Start with an empty set.
    # Add elements one by one.
    # When considering adding a new element x, check if S ∪ {x} is valid.
    # If not, try a different x.
    # This is slow because checking validity is O(m^2 * max_sum) with bitsets, but m is small.
    
    # Actually, let's try to construct a set that satisfies the stronger condition:
    # min_sum(k+1) > max_sum(k) for all k.
    # This ensures disjointness of sum sets.
    # Let S = {a_1, ..., a_m} with a_1 < a_2 < ... < a_m.
    # min_sum(k) = a_1 + ... + a_k
    # max_sum(k) = a_{m-k+1} + ... + a_m
    # Condition: a_1 + ... + a_{k+1} > a_{m-k+1} + ... + a_m for all k.
    
    # Let's try to find such a set.
    # For k=0: a_1 > a_m. Impossible unless m=1.
    # Wait, min_sum(1) = a_1, max_sum(0) = 0. Condition: a_1 > 0. True.
    # For k=1: min_sum(2) = a_1 + a_2, max_sum(1) = a_m. Condition: a_1 + a_2 > a_m.
    # For k=m-1: min_sum(m) = sum(S), max_sum(m-1) = sum(S) - a_1. Condition: sum(S) > sum(S) - a_1. True.
    
    # So the critical constraints are for intermediate k.
    # Let's try to maximize m subject to a_1 + ... + a_{k+1} > a_{m-k+1} + ... + a_m.
    
    # This is a different optimization problem. Let's see if we can find a large set this way.
    # If we choose a_i = i * d, then:
    # min_sum(k) = d * k(k+1)/2
    # max_sum(k) = d * (m(m+1)/2 - (m-k)(m-k+1)/2)
    # Condition: d * (k+1)(k+2)/2 > d * (m(m+1)/2 - (m-k)(m-k+1)/2)
    # (k+1)(k+2) > m(m+1) - (m-k)(m-k+1)
    # Let's test for m=60.
    # For k=30:
    # LHS = 31*32 = 992
    # RHS = 60*61 - 30*31 = 3660 - 930 = 2730
    # 992 > 2730 is False. So arithmetic progression won't work for m=60.
    
    # We need a "superincreasing" or similar sequence.
    # Let a_i grow exponentially.
    # a_i = 2^i.
    # min_sum(k) = 2^0 + ... + 2^{k-1} = 2^k - 1
    # max_sum(k) = 2^{m-k} + ... + 2^{m-1} = 2^m - 2^{m-k}
    # Condition: 2^k - 1 > 2^m - 2^{m-k}
    # For k=m/2, this requires 2^{m/2} > 2^m, which is false.
    
    # So even exponential doesn't work for large m with this strong condition.
    
    # Let's go back to the bitset DP and search.
    # We will use a local search:
    # 1. Start with a random set of size 60.
    # 2. While not valid:
    #    a. Find an element to remove.
    #    b. Remove it.
    # 3. Return the resulting set.
    
    # To find the element to remove, we can:
    # - Try removing each element and check validity. Keep the one that makes it valid.
    # - If none works, keep the one that reduces the number of collisions the most.
    
    # Counting collisions:
    # For each pair (i, j) with i < j, count the number of common sums between dp[i] and dp[j].
    # This is sum over all pairs of popcount(dp[i] & dp[j]).
    
    def count_collisions(S):
        m = len(S)
        if m == 0:
            return 0
        
        max_sum = sum(S)
        dp = [0] * (m + 1)
        dp[0] = 1
        
        for x in S:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << x)
        
        collisions = 0
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                collisions += bin(dp[i] & dp[j]).count('1')
        return collisions

    # This is expensive. Let's just try to find a valid set by random restarts.
    # Given the small size (60), maybe we can just try many random sets.
    
    # But wait, the seed returns 8. Can we do better?
    # Let's try to construct a set of size 10, 20, etc.
    
    # Let's try a greedy approach:
    # Start with empty set.
    # For each new element, try all candidates from 1..256, pick the one that keeps the set valid.
    # If no candidate works, stop.
    
    def greedy_construct():
        S = []
        used = set()
        for _ in range(60):
            found = False
            for x in range(1, 257):
                if x in used:
                    continue
                new_S = S + [x]
                if is_valid_fast(new_S):
                    S = new_S
                    used.add(x)
                    found = True
                    break
            if not found:
                break
        return S

    # This greedy approach might be slow because is_valid_fast is called many times.
    # But m is small, so it might be okay.
    
    # Let's try it.
    # Actually, let's optimize is_valid_fast by early termination.
    
    def is_valid_fast_early(S):
        m = len(S)
        if m == 0:
            return True
        
        dp = [0] * (m + 1)
        dp[0] = 1
        
        for x in S:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << x)
                # Early check: if dp[k] & dp[k-1] != 0, we have a collision?
                # No, collision is between different cardinalities.
                # We can check all pairs after each element? That's expensive.
        
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                if dp[i] & dp[j]:
                    return False
        return True

    # Let's run the greedy construction.
    S = greedy_construct()
    
    # If greedy doesn't reach 60, we can try local search to improve.
    # But for now, let's see what size we get.
    
    # To ensure we return a list of ints, we convert.
    return list(S)

# EVOLVE-BLOCK-END
