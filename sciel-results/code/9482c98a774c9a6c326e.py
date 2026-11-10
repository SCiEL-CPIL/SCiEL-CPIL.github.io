"""Construct a maximal set of distinct integers in 1..256 with cardinality-separated subset sums."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a set of distinct integers in 1..256 such that subset sums
    with different cardinalities are distinct.

    Strategy:
    1. Start with the known feasible AP {226, ..., 256} of size 31.
    2. Attempt to grow the set to size 32 by greedily adding elements
       while maintaining the cardinality separation property.
    3. Use a heuristic search: try adding the smallest feasible element
       that doesn't cause collisions, prioritizing elements that keep
       the sum ranges tight.
    
    If size 32 is not achievable, return the best found (size 31).
    """
    import random
    
    def is_valid_set(s):
        """Check if set s satisfies cardinality separation."""
        if len(s) == 0:
            return True
        s = sorted(s)
        n = len(s)
        # For each cardinality k, compute min and max subset sum
        # Min sum of k elements: sum of k smallest
        # Max sum of k elements: sum of k largest
        # Condition: min_sum[k] > max_sum[k-1] for all k=1..n
        # (empty set sum = 0, so min_sum[1] > 0 is automatic for positive ints)
        prefix_sums = [0]
        for x in s:
            prefix_sums.append(prefix_sums[-1] + x)
        
        # min_sum[k] = sum of k smallest = prefix_sums[k]
        # max_sum[k] = sum of k largest = prefix_sums[n] - prefix_sums[n-k]
        for k in range(1, n + 1):
            min_sum_k = prefix_sums[k]
            if k == 1:
                max_sum_prev = 0  # empty set
            else:
                max_sum_prev = prefix_sums[n] - prefix_sums[n - (k - 1)]
            if min_sum_k <= max_sum_prev:
                return False
        return True
    
    # Start with the AP solution
    current = list(range(226, 257))
    
    # Try to grow to size 32
    # We need to add one more element from 1..255 (since 226-256 are used)
    # But we might also need to modify existing elements
    
    # First, try simple addition: can we add any element from 1..255?
    candidates = list(range(1, 226))  # elements not in current set
    random.shuffle(candidates)
    
    best_size = len(current)
    best_set = current[:]
    
    for c in candidates:
        test_set = sorted(current + [c])
        if is_valid_set(test_set):
            best_size = len(test_set)
            best_set = test_set[:]
            break
    
    # If we couldn't grow by simple addition, try a more sophisticated search
    # Try replacing one element and adding another
    if best_size == 31:
        # Try: remove one element, add two new elements
        # This is expensive, so limit the search
        for i in range(len(current)):
            for c1 in range(1, 226):
                if c1 in current:
                    continue
                for c2 in range(c1 + 1, 226):
                    if c2 in current:
                        continue
                    test_set = sorted([x for j, x in enumerate(current) if j != i] + [c1, c2])
                    if is_valid_set(test_set):
                        best_size = len(test_set)
                        best_set = test_set[:]
                        break
                if best_size > 31:
                    break
            if best_size > 31:
                break
    
    # Return as list of Python ints
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
