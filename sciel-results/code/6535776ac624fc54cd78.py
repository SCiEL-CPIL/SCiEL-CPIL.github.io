"""Rewritten program: fast validity checks + multi-start greedy + local search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

DEADLINE = 80.0  # safety margin below the 90s wall-clock kill

def _is_valid_set(S):
    """Return True iff no element of S is the arithmetic mean of any subset of >=2 other elements."""
    n = len(S)
    if n < 3:
        return True
    S_sorted = sorted(S)
    # For each candidate a in S, check if a is the mean of some subset of S\{a}
    for a in S:
        others = [x for x in S if x != a]
        m = len(others)
        # We need to check if there exists a subset T of others with |T|>=2 such that sum(T) == |T| * a
        # This is equivalent to: does there exist k in [2..m] and a subset of size k with sum = k*a?
        # We can use a DP over the multiset of (x - a) for x in others.
        # Let d_i = x_i - a. We need a subset of indices with sum(d_i) = 0 and size >= 2.
        # But careful: we need the subset to be of OTHER elements, which it is by construction.
        # Also, the subset must have size >= 2.
        # 
        # DP: dp[s] = set of possible sizes of subsets of {d_0..d_{i}} that sum to s.
        # But s can range from -max_val to +max_val. max_val <= 512.
        # So s ranges from -512 to 512. That's 1025 values.
        # We can use a dict or a list.
        # 
        # Actually, a simpler approach: for each k from 2 to m, check if there's a subset of size k with sum = k*a.
        # This is a classic subset-sum with cardinality constraint.
        # 
        # For efficiency, let's use a bitset-based DP.
        # dp[i][j] = set of possible sums using i elements... but that's too slow.
        # 
        # Alternative: for each k, we can use a DP where dp[s] = max number of elements that can sum to s.
        # But we need exactly k elements.
        # 
        # Let's use: dp[j][s] = whether we can pick j elements from others that sum to s.
        # j ranges from 0 to m, s ranges from -512 to 512 (shifted by 512 to make non-negative).
        # 
        # Actually, let's think differently. The condition "a is the mean of a subset of others" 
        # means sum(T) = |T| * a for some T subset of others, |T| >= 2.
        # 
        # Let's just iterate over all subsets if m is small, or use DP if m is larger.
        # Since n <= 48, m <= 47. 2^47 is too large for brute force.
        # 
        # Let's use a DP approach:
        # For a given a, define d = [x - a for x in others].
        # We want to know if there exists a subset of d with sum 0 and size >= 2.
        # 
        # DP: let reachable[s] = set of sizes of subsets that sum to s.
        # Initialize reachable[0] = {0}.
        # For each d_i in d:
        #   new_reachable = copy of reachable
        #   for s, sizes in reachable.items():
        #     for sz in sizes:
        #       new_reachable[s + d_i].add(sz + 1)
        #   reachable = new_reachable
        # 
        # Then check if 0 in reachable and any size in reachable[0] is >= 2.
        # 
        # The range of s: each d_i is in [-511, 511], and there are at most 47 of them.
        # So s ranges from -47*511 to 47*511, which is about -24017 to 24017. That's ~48035 values.
        # This might be slow if we do this for each a.
        # 
        # Let's optimize: we only need to know if there's a subset of size >= 2 with sum 0.
        # We can track for each sum s, the minimum and maximum size of subsets achieving that sum.
        # Actually, we just need to know if any size >= 2 is achievable for sum 0.
        # 
        # Let's use: dp[s] = set of achievable sizes. But to save memory, we can use a bitset for sizes.
        # Sizes range from 0 to 47, so a 48-bit integer can represent the set of achievable sizes.
        # 
        # dp[s] = bitmask where bit j is set if a subset of size j sums to s.
        # 
        # Initialize: dp[0] = 1 (bit 0 set, meaning size 0 is achievable for sum 0).
        # 
        # For each d_i:
        #   new_dp = copy of dp
        #   for s, mask in dp.items():
        #     for j in range(m+1):
        #       if mask & (1 << j):
        #         new_dp[s + d_i] |= (1 << (j + 1))
        #   dp = new_dp
        # 
        # Then check if dp.get(0, 0) & ~1 != 0 (i.e., any bit other than bit 0 is set).
        # 
        # The number of distinct s values could be large, but in practice it's bounded.
        # Let's try this approach.

        d = [x - a for x in others]
        m = len(d)
        
        # dp: dict from sum to bitmask of achievable sizes
        dp = {0: 1}  # bit 0 set for sum 0
        
        for di in d:
            new_dp = dict(dp)
            for s, mask in dp.items():
                # For each size j that is achievable for sum s, size j+1 is achievable for sum s+di
                if mask:
                    # Iterate over set bits in mask
                    mm = mask
                    while mm:
                        lowbit = mm & (-mm)
                        j = lowbit.bit_length() - 1
                        new_mask = 1 << (j + 1)
                        new_dp[s + di] = new_dp.get(s + di, 0) | new_mask
                        mm ^= lowbit
            dp = new_dp
        
        # Check if sum 0 has any size >= 2 achievable
        if 0 in dp:
            mask = dp[0]
            if mask & ~1:  # any bit other than bit 0 is set
                return False
    
    return True


def _greedy_build(order, target_size=48):
    """Greedily build a valid set by adding elements in the given order."""
    selected = []
    for x in order:
        if len(selected) >= target_size:
            break
        candidate = selected + [x]
        if _is_valid_set(candidate):
            selected = candidate
    return selected


def _local_search(S, order, max_steps=2000):
    """Try to improve S by swapping elements with candidates from order."""
    S = list(S)
    n = len(S)
    for _ in range(max_steps):
        improved = False
        # Try to add a new element
        for x in order:
            if x not in S:
                candidate = S + [x]
                if len(candidate) <= 48 and _is_valid_set(candidate):
                    S = candidate
                    improved = True
                    break
        if improved:
            continue
        # Try to swap an element
        for i in range(len(S)):
            for x in order:
                if x not in S:
                    candidate = S[:i] + S[i+1:] + [x]
                    if len(candidate) == len(S) and _is_valid_set(candidate):
                        S = candidate
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    return S


def construct():
    start_time = time.time()
    best_set = []
    
    # Try multiple orderings
    orderings = []
    
    # Ordering 1: primes
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                return False
        return True
    primes = [x for x in range(1, 513) if is_prime(x)]
    orderings.append(primes)
    
    # Ordering 2: powers of 2
    powers = []
    p = 1
    while p <= 512:
        powers.append(p)
        p *= 2
    orderings.append(powers)
    
    # Ordering 3: random
    random.seed(42)
    rand_order = list(range(1, 513))
    random.shuffle(rand_order)
    orderings.append(rand_order)
    
    # Ordering 4: descending
    orderings.append(list(range(512, 0, -1)))
    
    # Ordering 5: ascending
    orderings.append(list(range(1, 513)))
    
    # Ordering 6: alternating small/large
    alt = []
    small = list(range(1, 257))
    large = list(range(512, 256, -1))
    for i in range(max(len(small), len(large))):
        if i < len(small):
            alt.append(small[i])
        if i < len(large):
            alt.append(large[i])
    orderings.append(alt)
    
    for order in orderings:
        if time.time() - start_time > DEADLINE - 5:
            break
        S = _greedy_build(order, 48)
        if len(S) > len(best_set):
            best_set = S
        # Try local search to improve
        if len(S) > 0:
            S2 = _local_search(S, order, max_steps=500)
            if len(S2) > len(best_set):
                best_set = S2
    
    # Final check
    if best_set and not _is_valid_set(best_set):
        # Fallback to a simple valid set
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
