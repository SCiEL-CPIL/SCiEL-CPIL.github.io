def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ. Maximize set size.
    
    Uses the condition: for each k >= 2, min k-sum > max (k-1)-sum,
    which is equivalent to: sum of k smallest > sum of (k-1) largest.
    """
    N = 256  # max value

    def check_set(s):
        """Check if sorted set s satisfies all cardinality separation constraints."""
        n = len(s)
        if n <= 1:
            return True
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + s[i]
        for k in range(2, n + 1):
            min_k_sum = prefix[k]
            max_km1_sum = prefix[n] - prefix[n - k + 1]
            if min_k_sum <= max_km1_sum:
                return False
        return True

    # Try n from 60 down to 1, using the n largest integers
    for n in range(60, 0, -1):
        s = list(range(257 - n, 257))  # {257-n, ..., 256}
        if check_set(s):
            return s

    # Fallback: return the seed
    return list(range(128, 136))
