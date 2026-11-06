"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s

    # Sieve primes up to 10000
    N = 10000
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for x in range(2, int(N ** 0.5) + 1):
        if is_prime[x]:
            is_prime[x * x::x] = bytearray(len(is_prime[x * x::x]))
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    n = len(primes)
    prime_index = {p: i for i, p in enumerate(primes)}
    
    # Precompute gaps between consecutive primes
    gaps = [primes[i+1] - primes[i] for i in range(n-1)]
    
    # DP: dp[i] = (length, last_prime, last_gap) for best sequence ending at prime i
    # We want to maximize length of sequence with non-decreasing gaps
    
    # Use memoized DFS or iterative DP
    # State: (index, min_gap) -> max length from index onwards with min_gap
    # But min_gap can be up to N, and index up to n, so state space is n * N which is too large
    
    # Alternative: DP where dp[i][j] = max length ending at prime i with last gap exactly j
    # But j can be up to N too
    
    # Better approach: Since we're looking for non-decreasing gaps, we can use a different DP
    # dp[i] = list of (last_gap, max_length) pairs for sequences ending at prime i
    # For each prime i, for each possible previous prime j < i, if gap = primes[i] - primes[j]
    # is >= last_gap in some sequence ending at j, we can extend it
    
    # This is still O(n^2 * avg_seq_length) which might be too slow
    
    # Let's try a simpler approach: greedy with backtracking or just a good heuristic
    
    # Actually, let's try dynamic programming where:
    # dp[i] = maximum length of valid sequence ending at prime i
    # To compute dp[i], we need to find j < i such that primes[i] - primes[j] >= gap_before_j
    # But we don't track gap_before_j directly
    
    # Let's define:
    # best_len[i] = max length of valid sequence ending at primes[i]
    # last_gap[i] = the gap that achieved best_len[i] (if multiple, pick the one that allows extension)
    
    # Actually, this doesn't work because a shorter sequence with smaller last gap might be better for extension
    
    # Let's use a different approach:
    # For each starting prime, try to build the longest sequence greedily
    # This is O(n^2) which is about 1e6, should be fast enough
    
    best_result = []
    
    for start_idx in range(n):
        if time.time() - t0 > DEADLINE:
            break
        seq = [primes[start_idx]]
        last_gap = 0
        current_idx = start_idx
        
        while True:
            # Find the next prime that maintains non-decreasing gap
            found = False
            for next_idx in range(current_idx + 1, n):
                gap = primes[next_idx] - primes[current_idx]
                if gap >= last_gap:
                    seq.append(primes[next_idx])
                    last_gap = gap
                    current_idx = next_idx
                    found = True
                    break
            
            if not found:
                break
        
        if len(seq) > len(best_result):
            best_result = seq[:]
    
    # Also try the original greedy approach as a baseline
    values = []
    for p in primes:
        if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
            values.append(p)
    
    if len(values) > len(best_result):
        best_result = values
    
    return best_result
# EVOLVE-BLOCK-END
