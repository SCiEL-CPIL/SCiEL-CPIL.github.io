"""Rewritten program: exact 0/1 knapsack via meet-in-the-middle for strict-underpacking."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    from fractions import Fraction
    from math import lcm

    deadline = time.time() + 55.0  # safety margin under 60s

    denominators = list(range(2, 49))
    n = len(denominators)

    # Compute LCM of all denominators to work in integer space
    L = 1
    for d in denominators:
        L = lcm(L, d)

    # Weights: w_i = L // d_i (so sum of selected weights < L means sum of reciprocals < 1)
    weights = np.array([L // d for d in denominators], dtype=np.int64)
    capacity = L - 1  # strict: sum of weights <= L-1

    # Split into two halves
    n1 = n // 2
    n2 = n - n1
    w1 = weights[:n1]
    w2 = weights[n1:]

    # Generate all subset sums for first half
    # n1 = 23, so 2^23 = 8388608 subsets
    def gen_subset_sums(w):
        m = len(w)
        # Use iterative approach: for each item, double the array
        sums = np.array([0], dtype=np.int64)
        for val in w:
            sums = np.concatenate([sums, sums + val])
        return sums

    # Generate subset sums and track which indices
    # For efficiency, we just need the sums, then reconstruct
    # But we need indices... let's store sums only first, then find best, then reconstruct

    # Actually, for meet-in-the-middle we need:
    # For each subset of first half with sum s1, find max s2 from second half such that s1 + s2 <= capacity
    # Then total = s1 + s2, deficit = L - total (minimize deficit = maximize total)

    # Step 1: Generate all subset sums for half 2, sort them
    if time.time() > deadline:
        # Fallback: greedy
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected

    sums2 = gen_subset_sums(w2)
    sums2_sorted = np.sort(sums2)

    # Step 2: For each subset sum s1 from half 1, binary search in sums2_sorted
    # for the largest value <= capacity - s1
    sums1 = gen_subset_sums(w1)

    best_total = 0
    best_s1 = 0
    best_s2 = 0

    # Vectorized binary search
    # For each s1, find the rightmost index in sums2_sorted where value <= capacity - s1
    targets = capacity - sums1
    # Use searchsorted to find insertion point, then take element before
    idx = np.searchsorted(sums2_sorted, targets, side='right') - 1
    # Filter valid (idx >= 0)
    valid = idx >= 0
    if np.any(valid):
        valid_idx = idx[valid]
        valid_s1 = sums1[valid]
        candidate_s2 = sums2_sorted[valid_idx]
        candidate_total = valid_s1 + candidate_s2
        # Find the max
        best_pos = np.argmax(candidate_total)
        best_total = int(candidate_total[best_pos])
        best_s1 = int(valid_s1[best_pos])
        best_s2 = int(candidate_s2[best_pos])

    if best_total == 0:
        # No valid subset found? Shouldn't happen since empty set has sum 0
        # But we need nonempty. Let's pick the single largest weight under capacity.
        # Actually empty set is sum=0 which is < L, but we need nonempty.
        # Let's just pick index 0 (denominator 2, weight L/2)
        return [0]

    # Now reconstruct: find which subset of half 1 gives sum best_s1
    # and which subset of half 2 gives sum best_s2

    def find_subset(w, target):
        """Find a subset of w that sums to exactly target. Returns list of indices."""
        m = len(w)
        # Use DFS with memoization or iterative
        # Since m is at most 23, we can use a simple approach
        # But 2^23 is too large for naive. Let's use the subset sums we already generated.
        # Actually, let's just do a simple backtracking since we know a solution exists.
        
        # Better: use the fact that we generated all sums. We can store the actual subset.
        # Let's regenerate with tracking.
        # For half 1 (23 elements), we can do a simple iterative DP-like approach.
        
        # Simple approach: try all subsets using bit manipulation for small m
        # For m=23, 2^23 = 8M, might be slow in Python. Let's use numpy.
        
        # Generate all subset sums with their index masks
        # Use a different approach: iterate and build up
        result = []
        remaining = list(w)
        # Greedy won't work for exact match. Let's use meet-in-the-middle again or just brute force for small side.
        
        # Since we already know the sum exists, let's do a simple recursive search with pruning
        # But to keep it fast, let's use the following:
        # For m <= 23, we can enumerate subsets using a stack-based approach
        
        # Actually, let's just do a simple DFS
        def dfs(start, current_sum, chosen):
            if current_sum == target:
                return chosen[:]
            if start >= m or current_sum > target:
                return None
            # Include
            r = dfs(start + 1, current_sum + w[start], chosen + [start])
            if r is not None:
                return r
            # Exclude
            r = dfs(start + 1, current_sum, chosen)
            if r is not None:
                return r
            return None
        
        return dfs(0, 0, [])

    # This DFS might be slow. Let's use a smarter approach.
    # Since we have the sorted sums2, we can find the specific subset by searching.
    # Alternative: regenerate subset sums with masks.
    
    # Let's use a different strategy: store the subset mask along with the sum.
    # For half 2 (22 elements), 2^22 = 4M, manageable.
    # For half 1 (23 elements), 2^23 = 8M, also manageable with numpy.
    
    # Let's redo with masks.
    def gen_subset_sums_with_masks(w):
        m = len(w)
        size = 1 << m
        sums = np.zeros(size, dtype=np.int64)
        # Compute all subset sums
        for i in range(m):
            bit = 1 << i
            val = w[i]
            # For all masks with this bit set, add val
            # This is a standard technique
            for mask in range(size):
                if mask & bit:
                    sums[mask] += val
        return sums

    # This is O(m * 2^m) which for m=23 is 23 * 8M = 184M operations, too slow in Python.
    
    # Let's use the iterative doubling approach which is O(2^m) total.
    def gen_subset_sums_iterative(w):
        m = len(w)
        sums = np.array([0], dtype=np.int64)
        for val in w:
            sums = np.concatenate([sums, sums + val])
        return sums

    # We already have sums1 and sums2. We need to find WHICH subset gives the target sum.
    # Let's use a simple approach: for the half with the target, do a backward reconstruction.
    
    def find_subset_by_sum(w, target):
        """Find subset of w summing to target. Returns list of indices into w."""
        m = len(w)
        # Work backwards
        # Generate all prefix sums
        # Use the iterative method but track the path
        # Simple approach: since m <= 23, use a set-based DP
        # dp[s] = last index that achieved sum s
        # But sums can be large. Let's just do a simple recursive search with memoization.
        
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def solve(i, remaining):
            if remaining == 0:
                return []
            if i == m or remaining < 0:
                return None
            # Try including w[i]
            if remaining >= w[i]:
                r = solve(i + 1, remaining - w[i])
                if r is not None:
                    return [i] + r
            # Try excluding w[i]
            r = solve(i + 1, remaining)
            if r is not None:
                return r
            return None
        
        result = solve(0, target)
        if result is None:
            # Should not happen
            return [0]
        return result

    subset1 = find_subset_by_sum(w1, best_s1)
    subset2 = find_subset_by_sum(w2, best_s2)

    # Convert to original indices
    selected = list(subset1) + [n1 + i for i in subset2]

    # Ensure nonempty
    if not selected:
        selected = [0]

    return selected

# EVOLVE-BLOCK-END
