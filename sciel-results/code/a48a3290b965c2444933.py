"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import isqrt
    
    # Generate all primes <= 10000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(10000)
    n = len(primes)
    
    if n <= 2:
        return primes

    # dp[i] is a list of length n - 1 - i
    # dp[i][k] corresponds to dp[i][i + 1 + k] in the full 2D notation
    # i.e., dp[i][j] where j = i + 1 + k
    dp = [None] * n
    parent = [None] * n
    
    best_len = 2
    best_pair = (0, 1)
    
    # Initialize dp for i = 0
    # dp[0][j] = 2 for all j > 0
    if n > 1:
        dp[0] = [2] * (n - 1)
        parent[0] = [-1] * (n - 1)
        
    for i in range(1, n - 1):
        # We need dp[h][i] for all h < i
        # dp[h][i] is stored at dp[h][i - h - 1]
        
        # Build list of (gap, dp_val, h) for h = i-1 down to 0
        # gap = primes[i] - primes[h]
        # As h decreases, gap increases.
        # We want to process h such that gap is increasing?
        # The condition is gap_h <= req_gap.
        # As j increases, req_gap increases.
        # We want to add more h's to the "valid" set.
        # If we sort h by gap ascending, we can use a pointer.
        # gap_h = primes[i] - primes[h].
        # h = i-1 -> gap = primes[i] - primes[i-1] (smallest)
        # h = 0   -> gap = primes[i] - primes[0] (largest)
        # So h in descending order (i-1, i-2, ..., 0) gives gap in ascending order.
        
        sorted_h = list(range(i - 1, -1, -1))
        gaps = [primes[i] - primes[h] for h in sorted_h]
        dp_vals = [dp[h][i - h - 1] for h in sorted_h]
        
        dp[i] = [2] * (n - 1 - i)
        parent[i] = [-1] * (n - 1 - i)
        
        ptr = 0
        current_max = 0
        best_h = -1
        
        for j in range(i + 1, n):
            req_gap = primes[j] - primes[i]
            
            # Advance pointer while gap is within limit
            while ptr < i and gaps[ptr] <= req_gap:
                if dp_vals[ptr] > current_max:
                    current_max = dp_vals[ptr]
                    best_h = sorted_h[ptr]
                ptr += 1
            
            if current_max > 0:
                val = 2 + current_max
                p_val = best_h
            else:
                val = 2
                p_val = -1
                
            dp[i][j - i - 1] = val
            parent[i][j - i - 1] = p_val
            
            if val > best_len:
                best_len = val
                best_pair = (i, j)
                
    # If n is small, handle case where loop didn't run or best_len is still 2
    if best_len == 2:
        return [primes[0], primes[1]]
        
    # Backtrack
    # best_pair is (i, j)
    # We need to reconstruct the sequence ending with primes[i], primes[j]
    # parent[i][j-i-1] gives the previous index h
    # Then we go to (h, i)
    
    seq = []
    i, j = best_pair
    seq.append(primes[j])
    
    while i > 0:
        h = parent[i][j - i - 1]
        if h == -1:
            break
        seq.append(primes[i])
        j = i
        i = h
        
    seq.append(primes[i])
    seq.reverse()
    
    return seq
# EVOLVE-BLOCK-END
