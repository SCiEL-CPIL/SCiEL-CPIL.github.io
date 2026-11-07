"""Program for cardinality-separated subset sums using reverse greedy construction."""
# EVOLVE-BLOCK-START

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ. Maximize set size.
    
    Uses reverse greedy: start with max cardinality, assign largest available values
    while maintaining min k-sum > max (k-1)-sum for all k.
    """
    
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
    
    def try_consecutive(n):
        """Try the n largest consecutive integers {257-n, ..., 256}."""
        s = list(range(257 - n, 257))
        if check_set(s):
            return s
        return None
    
    def try_greedy_with_gaps(n):
        """
        Try to build a set of size n using a greedy approach that allows gaps.
        We assign values to positions 0..n-1 (sorted order), trying to maximize
        the minimum element while keeping the maximum at 256.
        
        Strategy: Start with all positions at 256 (max). Then iteratively reduce
        positions from the bottom up, choosing the largest value that still allows
        the constraints to be satisfiable.
        
        Actually, a better strategy: work from the top down. Set s[n-1]=256, s[n-2]=255, etc.
        Then for the bottom positions, try to find values that satisfy constraints.
        
        Even better: use a parametric approach. Assume s[i] = base + i*step for i=0..n-1,
        but allow some flexibility. Or just do a direct greedy.
        
        Let's try: fix the top half to be as large as possible (256, 255, ...),
        and solve for the bottom half.
        """
        # We'll try a binary search on the minimum element, and for each min,
        # try to fill in the rest greedily.
        
        # Actually, let's try a direct construction:
        # For a given n, we need s[0] < s[1] < ... < s[n-1] <= 256
        # with s[0]+...+s[k-1] > s[n-k+1]+...+s[n-1] for all k=2..n
        
        # The tightest constraints are usually in the middle.
        # Let's try: s[i] = 256 - (n-1-i) for i = 0..n-1 (i.e., the n largest)
        # If that fails, we need to make the bottom elements larger relative to the top.
        # But we can't make them larger than the top... 
        # 
        # Wait, we CAN use non-consecutive values. For example, we could skip some large
        # values and use more medium values. But that would make the min k-sums smaller.
        #
        # Actually, the issue is that for the n largest consecutive integers, the bottom
        # elements are too small relative to the top. We need the bottom elements to be
        # larger. The only way to do that is to NOT use the very largest values, and instead
        # use values that are more clustered.
        #
        # Let's try: use values in a range [L, L+n-1] where L is chosen to satisfy constraints.
        # For consecutive values in [L, L+n-1]:
        # s[i] = L + i
        # min k-sum = k*L + k(k-1)/2
        # max (k-1)-sum = (k-1)*(L+n-1) - (k-1)(k-2)/2
        # Need: k*L + k(k-1)/2 > (k-1)*(L+n-1) - (k-1)(k-2)/2
        # k*L + k(k-1)/2 > (k-1)*L + (k-1)(n-1) - (k-1)(k-2)/2
        # L > (k-1)(n-1) - (k-1)(k-2)/2 - k(k-1)/2
        # L > (k-1)[(n-1) - (k-2)/2 - k/2]
        # L > (k-1)[n-1 - (k-2+k)/2]
        # L > (k-1)[n-1 - (2k-2)/2]
        # L > (k-1)[n-1 - (k-1)]
        # L > (k-1)(n-k)
        #
        # So we need L > max over k=2..n of (k-1)(n-k)
        # The maximum of (k-1)(n-k) occurs at k = (n+1)/2 (approximately).
        # For k = (n+1)/2: (k-1)(n-k) = ((n-1)/2)((n-1)/2) = (n-1)^2/4
        #
        # So we need L > (n-1)^2/4, and L+n-1 <= 256, so L <= 257-n.
        # Need: (n-1)^2/4 < 257-n
        # (n-1)^2 < 4(257-n)
        # n^2 - 2n + 1 < 1028 - 4n
        # n^2 + 2n - 1027 < 0
        # n < (-2 + sqrt(4 + 4*1027))/2 = (-2 + sqrt(4112))/2 ≈ (-2 + 64.13)/2 ≈ 31.06
        #
        # So for consecutive values, n can be at most 31.
        # Let's verify: for n=31, (n-1)^2/4 = 900/4 = 225. L > 225, so L >= 226.
        # L+n-1 = 226+30 = 256. OK!
        # For n=32: (31)^2/4 = 961/4 = 240.25. L > 240.25, so L >= 241.
        # L+n-1 = 241+31 = 272 > 256. Fails.
        #
        # So with consecutive values, max n is 31.
        # But we can do better with non-consecutive values!
        
        # Let's try a different approach: use a geometric or other sequence.
        # Or: use a greedy that doesn't require consecutive values.
        
        # Greedy approach for fixed n:
        # Start with s[n-1] = 256, s[n-2] = 255, ..., and work backwards.
        # At each step, choose the largest possible value that doesn't violate
        # any constraint that is already fully determined.
        
        # Actually, let's just try: for n from 60 down, try the n largest consecutive.
        # If that fails, try n-1, etc. We already know from the formula that
        # consecutive values work up to n=31.
        
        # For n > 31, we need a smarter construction. Let's try using values
        # that are not consecutive but are chosen to satisfy the constraints.
        
        # One idea: use s[i] = 256 - floor(i * (256 - L) / (n-1)) for some L.
        # This gives a roughly linear spacing.
        
        # Actually, let's just do a direct search for each n:
        # Try to find any set of n distinct integers in 1..256 that satisfies the constraints.
        # We can use a greedy: assign values from top to bottom, choosing the largest
        # available value at each step that doesn't make the problem infeasible.
        
        # A simpler approach: just try consecutive sets for n from 60 down to 1.
        # We know n=31 works with consecutive. Let's see if we can do better.
        
        return None
    
    # First, try consecutive sets from n=60 down to 1
    for n in range(60, 0, -1):
        result = try_consecutive(n)
        if result is not None:
            return result
    
    # Fallback: return the seed
    return list(range(128, 136))

# EVOLVE-BLOCK-END
