import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    recips = [Fraction(1, d) for d in denominators]
    
    # Meet-in-the-Middle approach
    # Split indices into two halves
    mid = n // 2
    left_indices = list(range(mid))
    right_indices = list(range(mid, n))
    
    n_left = len(left_indices)
    n_right = len(right_indices)
    
    # Generate all subsets of left half with their sums
    # For each subset, store (sum, indices_tuple)
    # We want sums strictly less than 1
    left_sums = []
    for mask in range(1 << n_left):
        s = Fraction(0)
        indices = []
        valid = True
        for i in range(n_left):
            if mask & (1 << i):
                s += recips[left_indices[i]]
                indices.append(left_indices[i])
        if s < 1:
            left_sums.append((s, tuple(indices)))
    
    # Sort left_sums by sum descending (largest sums first)
    left_sums.sort(key=lambda x: x[0], reverse=True)
    
    # Generate all subsets of right half with their sums
    right_sums = []
    for mask in range(1 << n_right):
        s = Fraction(0)
        indices = []
        for i in range(n_right):
            if mask & (1 << i):
                s += recips[right_indices[i]]
                indices.append(right_indices[i])
        if s < 1:
            right_sums.append((s, tuple(indices)))
    
    # Sort right_sums by sum descending
    right_sums.sort(key=lambda x: x[0], reverse=True)
    
    best_deficit = Fraction(1, 1)
    best_indices = []
    
    # For each left subset, find the best right subset such that total < 1
    # Since we want to minimize 1 - (left_sum + right_sum), we want to maximize left_sum + right_sum subject to < 1
    
    # For a given left_sum, we need the largest right_sum such that left_sum + right_sum < 1
    # i.e., right_sum < 1 - left_sum
    
    # Since right_sums is sorted descending, we can binary search or linear scan
    # But with ~131k entries on each side, a two-pointer approach might work
    
    # Actually, let's use a different approach:
    # For each left entry, binary search in right_sums for the largest sum < 1 - left_sum
    
    import bisect
    
    # Extract just the sums for binary search
    right_sum_values = [x[0] for x in right_sums]
    
    # We want the largest value in right_sum_values that is < 1 - left_sum
    # Since right_sum_values is sorted descending, we need to find the rightmost position where value < target
    # Let's reverse to ascending for standard bisect
    right_sum_values_asc = list(reversed(right_sum_values))
    # And we need corresponding indices
    # Actually, let's just iterate through right_sums with a pointer
    
    # Better: for each left entry, scan right_sums from the end (smallest sums) to find... no
    # Let me think differently.
    
    # right_sums is sorted descending. For a target T = 1 - left_sum, we want the largest right_sum < T.
    # Since sorted descending, we find the first element from the left that is < T. That's the largest such.
    
    # We can binary search for this.
    
    # Create ascending version for bisect
    # right_sum_values is descending. Let's make ascending.
    # Actually, let's just do a linear scan with two pointers since both are sorted.
    
    # Alternative: for each left entry, do a binary search.
    # n_left subsets: up to 2^23 = 8.4M, n_right subsets: up to 2^24 = 16.7M
    # That's too many to store all at once potentially.
    
    # Let's check sizes
    # n_left = 23, n_right = 24
    # 2^23 = 8388608, 2^24 = 16777216
    # Storing these as Fractions might be memory intensive but let's try.
    
    # Actually, the current code already generates them. Let's proceed with binary search.
    
    # For binary search in descending array for largest value < target:
    # We want the index of the first element that is < target.
    
    def find_best_right(target):
        """Find the entry in right_sums with the largest sum < target.
        right_sums is sorted by sum descending."""
        # Binary search for the first element where sum < target
        lo = 0
        hi = len(right_sums)
        while lo < hi:
            mid = (lo + hi) // 2
            if right_sums[mid][0] < target:
                hi = mid
            else:
                lo = mid + 1
        if lo < len(right_sums):
            return right_sums[lo]
        return None
    
    # Iterate through all left subsets
    for left_sum, left_idx in left_sums:
        if time.time() - start_time > DEADLINE:
            break
        
        target = 1 - left_sum
        # target is positive since left_sum < 1
        best_right = find_best_right(target)
        
        if best_right is not None:
            total_sum = left_sum + best_right[0]
            if total_sum < 1:
                deficit = 1 - total_sum
                if deficit < best_deficit:
                    best_deficit = deficit
                    combined = list(left_idx) + list(best_right[1])
                    best_indices = combined
    
    # Also consider cases where only left or only right is used
    # These are already covered since the other side can be empty (sum=0)
    # Empty subset has sum 0, which is < 1, so it's in both lists
    
    if not best_indices:
        # Fallback: just pick index 0
        best_indices = [0]
    
    return best_indices
