import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ.
    
    First tries the known 31-element AP [226..256].
    Then attempts to find a larger valid set via greedy search with fast validation.
    """
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    # Known valid 31-element solution
    fallback = list(range(226, 257))
    
    # Quick validation helper: check if a set satisfies the cardinality-separated property
    def is_valid(s):
        n = len(s)
        if n == 0:
            return True
        # Compute min and max sum for each cardinality
        s_sorted = sorted(s)
        # Prefix sums for min sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + s_sorted[i]
        # min_sum[k] = prefix[k], max_sum[k] = prefix[n] - prefix[n-k]
        min_sums = [prefix[k] for k in range(n + 1)]
        max_sums = [prefix[n] - prefix[n - k] for k in range(n + 1)]
        # Check disjointness: for c1 < c2, need max_sums[c1] < min_sums[c2]
        for c1 in range(n + 1):
            for c2 in range(c1 + 1, n + 1):
                if max_sums[c1] >= min_sums[c2]:
                    return False
        return True
    
    # Try to find a 32-element set via greedy construction
    # Strategy: try different starting points and spacings
    # For an AP of length n starting at a: elements are a, a+d, a+2d, ..., a+(n-1)d
    # We need a + (n-1)*d <= 256 and a >= 1
    
    # Try APs with various parameters
    best_size = 31
    best_set = fallback
    
    # Try to find 32+ element APs
    for n in range(32, 40):
        if time.time() > deadline:
            break
        for d in range(1, 4):
            # a + (n-1)*d <= 256 => a <= 256 - (n-1)*d
            # a >= 1
            max_a = 256 - (n - 1) * d
            if max_a < 1:
                continue
            # Try a few starting points
            for a in [1, max_a, max_a - d, max_a - 2*d]:
                if a < 1 or a > max_a:
                    continue
                s = [a + i * d for i in range(n)]
                if s[-1] > 256:
                    continue
                if is_valid(s):
                    if n > best_size:
                        best_size = n
                        best_set = s
                    break
            if best_size >= n:
                break
    
    # Also try a greedy non-AP approach for 32 elements
    if best_size < 32 and time.time() < deadline - 10:
        # Greedy: try to build a 32-element set
        # Start with the 31-element solution and try to add/modify elements
        # Alternative: try sets with elements clustered at both ends
        # For cardinality separation, we want small elements to have small sums
        # and large elements to have large sums
        
        # Try: mix of small and large elements
        # Key insight: if we have elements that are "far apart", the sum ranges
        # for different cardinalities will be more separated
        
        # Try a geometric-like progression
        for base in [2, 3]:
            if time.time() > deadline:
                break
            # Generate candidates: base^0, base^1, ..., but capped at 256
            candidates = []
            v = 1
            while v <= 256:
                candidates.append(v)
                v = v * base + 1  # slightly more than geometric
            # We need 32 distinct elements
            if len(candidates) >= 32:
                # Try the first 32
                s = candidates[:32]
                if is_valid(s):
                    if 32 > best_size:
                        best_size = 32
                        best_set = s
                # Try the last 32
                s = candidates[-32:]
                if is_valid(s):
                    if 32 > best_size:
                        best_size = 32
                        best_set = s
    
    # Try a more systematic search for 32 elements
    if best_size < 32 and time.time() < deadline - 5:
        # Use the insight that for the AP to work, we need the gap between
        # max_sum(k) and min_sum(k+1) to be positive for all k.
        # For a set sorted as x_1 < x_2 < ... < x_n:
        # min_sum(k+1) - max_sum(k) = sum(x_1..x_{k+1}) - sum(x_{n-k}..x_n)
        # This is hard to control directly, so let's try a different approach.
        
        # Try: take the 31-element AP and try to insert one more element
        # or shift elements to make room
        base_set = sorted(range(226, 257))
        
        # Try shifting: use range(a, a+32) for various a
        for a in range(1, 225):
            if time.time() > deadline:
                break
            s = list(range(a, a + 32))
            if s[-1] > 256:
                break
            if is_valid(s):
                best_size = 32
                best_set = s
                break
        
        # If no contiguous 32-element AP works, try non-contiguous
        if best_size < 32:
            # Try: 16 small elements + 16 large elements
            # Small: 1..16, Large: 241..256
            s_small = list(range(1, 17))
            s_large = list(range(241, 257))
            s = s_small + s_large
            if len(set(s)) == 32 and is_valid(s):
                best_size = 32
                best_set = s
            else:
                # Try various splits
                for n_small in range(8, 24):
                    if time.time() > deadline:
                        break
                    n_large = 32 - n_small
                    s_small = list(range(1, n_small + 1))
                    s_large = list(range(257 - n_large, 257))
                    s = s_small + s_large
                    if len(set(s)) == 32 and is_valid(s):
                        best_size = 32
                        best_set = s
                        break
    
    # Final safety check
    if not is_valid(best_set):
        return fallback
    
    # Ensure return type is correct
    result = [int(x) for x in best_set]
    return result
# EVOLVE-BLOCK-END
