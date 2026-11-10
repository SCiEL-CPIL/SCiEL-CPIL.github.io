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
    
    # dp[j][i] = length of longest valid subsequence ending with primes[i], primes[j] (i < j)
    # parent[j][i] = the index h < i that achieved the max for dp[j][i]
    
    # Initialize dp and parent
    # dp[j] is a list of length j, containing dp[j][0]...dp[j][j-1]
    dp = [None] * n
    parent = [None] * n
    
    for j in range(n):
        dp[j] = [2] * j
        parent[j] = [-1] * j
    
    best_len = 2
    best_pair = (0, 1)
    
    # Iterate mid from 1 to n-2
    # mid is the second-to-last element index
    for mid in range(1, n - 1):
        vals = dp[mid]
        # vals[k] is dp[mid][k] for k < mid
        
        # We want to compute dp[last][mid] for last > mid
        # dp[last][mid] = 2 + max{ dp[mid][prev] : prev < mid, p[mid] - p[prev] <= p[last] - p[mid] }
        
        # Condition: p[prev] >= 2*p[mid] - p[last]
        
        # As last increases, p[last] increases, so threshold T = 2*p[mid] - p[last] decreases.
        # Valid prevs are those with p[prev] >= T.
        # Since p is increasing, valid prevs form a suffix [k_min, mid-1].
        # We maintain ptr = k_min and current_max = max(vals[ptr...mid-1]).
        
        ptr = mid
        current_max = 0
        best_prev = -1
        
        for last in range(mid + 1, n):
            T = 2 * primes[mid] - primes[last]
            
            # Expand ptr to the left while p[ptr-1] >= T
            while ptr > 0 and primes[ptr - 1] >= T:
                ptr -= 1
                if vals[ptr] > current_max:
                    current_max = vals[ptr]
                    best_prev = ptr
            
            length = 2 + current_max
            dp[last][mid] = length
            parent[last][mid] = best_prev
            
            if length > best_len:
                best_len = length
                best_pair = (mid, last)
    
    # Backtrack
    if best_len == 2:
        return [primes[0], primes[1]]
        
    seq = []
    i, j = best_pair
    seq.append(primes[j])
    while i != -1:
        seq.append(primes[i])
        prev = parent[j][i]
        if prev == -1:
            break
        j = i
        i = prev
        
    seq.reverse()
    return seq
# EVOLVE-BLOCK-END
