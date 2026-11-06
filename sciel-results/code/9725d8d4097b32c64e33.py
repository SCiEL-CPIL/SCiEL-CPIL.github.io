"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import bisect

def _sieve_primes(limit):
    """Return list of primes <= limit."""
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    deadline = time.time() + 55  # safety margin before 60s kill
    primes = _sieve_primes(10000)
    n = len(primes)
    
    if n < 2:
        return primes
    
    # dp[i][j] = longest valid subsequence ending with primes[i], primes[j] (i < j)
    # parent[i][j] = the index h < i that gives the best extension to (i,j)
    
    dp = [[2] * n for _ in range(n)]
    parent = [[-1] * n for _ in range(n)]
    
    # Process i from 1 to n-1 (i=0 has no h < 0)
    for i in range(1, n):
        if time.time() > deadline:
            break
        # Collect (gap, h, dp[h][i]) for h < i
        pairs = []
        for h in range(i):
            gap = primes[i] - primes[h]
            pairs.append((gap, h, dp[h][i]))
        # Sort by gap
        pairs.sort()
        # Compute prefix max and track which h gives it
        prefix_max = []
        prefix_arg = []
        cur_max = 0
        cur_arg = -1
        for gap, h, val in pairs:
            if val > cur_max:
                cur_max = val
                cur_arg = h
            prefix_max.append(cur_max)
            prefix_arg.append(cur_arg)
        
        gaps_sorted = [p[0] for p in pairs]
        
        for j in range(i + 1, n):
            max_gap = primes[j] - primes[i]
            # Find rightmost index where gap <= max_gap
            idx = bisect.bisect_right(gaps_sorted, max_gap) - 1
            if idx >= 0:
                candidate = 1 + prefix_max[idx]
                if candidate > dp[i][j]:
                    dp[i][j] = candidate
                    parent[i][j] = prefix_arg[idx]
    
    # Find the best pair (i, j) maximizing dp[i][j]
    best_len = 2
    best_i = 0
    best_j = 1
    for i in range(n):
        for j in range(i + 1, n):
            if dp[i][j] > best_len:
                best_len = dp[i][j]
                best_i = i
                best_j = j
    
    if best_len <= 2:
        return [primes[best_i], primes[best_j]]
    
    # Reconstruct the sequence by following parent pointers
    seq = []
    i, j = best_i, best_j
    while True:
        seq.append(primes[j])
        h = parent[i][j]
        if h == -1:
            # This was a base case (length 2), so add the previous element
            seq.append(primes[i])
            break
        j = i
        i = h
    seq.reverse()
    
    return seq

# EVOLVE-BLOCK-END
