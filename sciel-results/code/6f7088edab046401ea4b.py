"""Closest strict underpacking of one by supplied unit fractions."""
# EVOLVE-BLOCK-START
import time

def construct():
    import numpy as np
    from math import gcd
    from fractions import Fraction

    deadline = time.time() + 80.0  # safety margin under 90s limit

    denominators = list(range(2, 49))  # 47 denominators
    n = len(denominators)

    # Split into two halves
    mid = n // 2  # 23
    first = denominators[:mid]    # 23 denominators
    second = denominators[mid:]   # 24 denominators

    # Compute LCM of first half
    L1 = 1
    for d in first:
        L1 = L1 * d // gcd(L1, d)

    # Compute LCM of second half
    L2 = 1
    for d in second:
        L2 = L2 * d // gcd(L2, d)

    # Common LCM
    L = L1 * L2 // gcd(L1, L2)

    # Numerators: L / d for each denominator
    # First half: L / d for d in first
    nums1 = np.array([L // d for d in first], dtype=np.int64)
    # Second half: L / d for d in second
    nums2 = np.array([L // d for d in second], dtype=np.int64)

    # Generate all subset sums for first half using iterative approach
    # 2^23 = 8,388,608 subsets
    def gen_subset_sums(nums):
        """Generate all subset sums of nums using vectorized shifts."""
        n = len(nums)
        # Start with {0}
        sums = np.array([0], dtype=np.int64)
        for i in range(n):
            if time.time() > deadline - 10.0:
                break  # safety: stop adding more
            # Add nums[i] to all existing sums
            new_sums = sums + nums[i]
            sums = np.concatenate([sums, new_sums])
        return sums

    t0 = time.time()
    sums1 = gen_subset_sums(nums1)
    t1 = time.time()
    sums2 = gen_subset_sums(nums2)
    t2 = time.time()

    # Sort sums2 for binary search
    sums2_sorted = np.sort(sums2)

    # We need sum1 + sum2 < L (strictly)
    # For each sum1, find max sum2 such that sum1 + sum2 < L
    # i.e., sum2 < L - sum1
    # Using searchsorted: find rightmost index where sums2_sorted < L - sum1
    # np.searchsorted(sums2_sorted, L - sum1 - 1, side='right') gives count of elements <= L-sum1-1
    # Actually we want the largest sum2 where sum2 <= L - sum1 - 1
    # searchsorted(sums2_sorted, L - sum1 - 1, side='right') returns index after last element <= L-sum1-1
    # So the element at that index - 1 is the largest one

    # Vectorized: for each s1, target = L - s1 - 1
    targets = L - sums1 - 1  # we want largest s2 <= targets[i]

    # Filter out targets that are negative (no valid s2)
    valid_mask = targets >= 0
    if not np.any(valid_mask):
        # No valid combination found; fall back to greedy
        selected = []
        total = Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected

    # For valid ones, binary search
    indices = np.searchsorted(sums2_sorted, targets[valid_mask], side='right') - 1
    # indices[i] is the index in sums2_sorted of the largest s2 <= targets[i]
    # But we need to ensure indices[i] >= 0
    valid2 = indices >= 0
    if not np.any(valid2):
        selected = []
        total = Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected

    best_s1_indices = np.where(valid_mask)[0][valid2]
    best_s2_indices = indices[valid2]
    best_s2_values = sums2_sorted[best_s2_indices]

    # Compute total for each candidate
    total_values = sums1[best_s1_indices] + best_s2_values

    # Find the maximum total (closest to L but strictly below)
    max_total_idx = np.argmax(total_values)
    best_s1_val = sums1[best_s1_indices[max_total_idx]]
    best_s2_val = best_s2_values[max_total_idx]

    # Reconstruct the subset from best_s1_val
    # We need to find which subset of nums1 sums to best_s1_val
    # And which subset of nums2 sums to best_s2_val

    def find_subset(nums, target, max_iter=100000):
        """Find a subset of nums that sums to target using DP/bitset."""
        n = len(nums)
        # Use a dictionary: sum -> set of indices (or just track one path)
        # For efficiency, use a set of reachable sums and track predecessors
        # But for 23/24 elements, we can use a different approach:
        # Greedy/backtrack on sorted nums
        
        # Sort nums with original indices
        indexed = sorted(enumerate(nums), key=lambda x: -x[1])  # descending
        
        # Backtracking search
        def backtrack(idx, remaining, chosen):
            if remaining == 0:
                return chosen
            if idx >= len(indexed):
                return None
            # Try including
            orig_idx, val = indexed[idx]
            if val <= remaining:
                result = backtrack(idx + 1, remaining - val, chosen + [orig_idx])
                if result is not None:
                    return result
            # Try excluding
            return backtrack(idx + 1, remaining, chosen)
        
        result = backtrack(0, target, [])
        return result

    s1_subset = find_subset(nums1, int(best_s1_val))
    if s1_subset is None:
        # Fallback: use greedy
        selected = []
        total = Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected

    s2_subset = find_subset(nums2, int(best_s2_val))
    if s2_subset is None:
        # Fallback
        selected = []
        total = Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected

    # Map back to original indices
    # s1_subset contains indices into nums1 (which correspond to first half)
    # s2_subset contains indices into nums2 (which correspond to second half, offset by mid)
    selected = []
    for idx in s1_subset:
        selected.append(idx)  # idx is already 0-based in first half
    for idx in s2_subset:
        selected.append(mid + idx)  # offset for second half

    # Verify: must be nonempty, distinct, sum strictly below 1
    if not selected:
        # Fallback to greedy
        total = Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected

    # Verify distinctness
    if len(set(selected)) != len(selected):
        # Fallback
        total = Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected

    # Verify sum < 1
    total = Fraction()
    for idx in selected:
        total += Fraction(1, denominators[idx])
    if total >= 1:
        # Fallback
        total = Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        return selected

    return sorted(selected)

# EVOLVE-BLOCK-END
