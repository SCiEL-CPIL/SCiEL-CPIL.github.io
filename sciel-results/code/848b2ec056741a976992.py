"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import bisect

def _sieve(n):
    """Return list of primes <= n using Sieve of Eratosthenes."""
    if n < 2:
        return []
    is_prime = bytearray(b'\x01') * (n + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    primes = _sieve(10000)
    n = len(primes)
    
    # dp[i] = length of longest valid sequence ending at primes[i]
    # parent[i] = index of previous element in the sequence
    # last_gap[i] = gap between primes[i] and primes[parent[i]]
    dp = [1] * n
    parent = [-1] * n
    last_gap = [0] * n  # gap into the current element (0 for single element)
    
    best_idx = 0
    best_len = 1
    
    # For each pair (j, i) with j < i, check if we can extend sequence ending at j to i
    # Condition: primes[i] - primes[j] >= last_gap[j]
    # This is O(n^2) but n ~ 1229, so ~750k pairs - very fast
    
    # Optimization: for each i, we want to find the best j < i such that:
    #   primes[i] - primes[j] >= last_gap[j]
    #   and dp[j] is maximized
    # 
    # We can iterate j from i-1 down to 0, but that's still O(n^2).
    # Let's just do the straightforward O(n^2) since n is small.
    
    for i in range(1, n):
        pi = primes[i]
        # Check time
        if time.time() > deadline:
            break
        for j in range(i - 1, -1, -1):
            gap = pi - primes[j]
            if gap >= last_gap[j]:
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
                    last_gap[i] = gap
                # Since we iterate j from high to low, and gaps increase as j decreases,
                # once gap >= last_gap[j], larger j (smaller gap) won't help unless dp[j] is bigger.
                # Actually, as j decreases, gap increases, so the condition becomes easier to satisfy.
                # But dp[j] varies. We need to check all valid j.
                # However, we can break early if even the maximum possible dp[j]+1 can't beat dp[i].
                # The max possible is (i) since at most i elements before i.
                # This heuristic may not help much. Let's just continue.
    
    # Find the best ending
    for i in range(n):
        if dp[i] > best_len:
            best_len = dp[i]
            best_idx = i
    
    # Reconstruct the sequence
    seq = []
    idx = best_idx
    while idx != -1:
        seq.append(primes[idx])
        idx = parent[idx]
    seq.reverse()
    
    return seq
# EVOLVE-BLOCK-END
