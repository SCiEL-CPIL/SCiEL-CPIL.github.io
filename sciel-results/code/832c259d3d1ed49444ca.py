import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Sieve of Eratosthenes to get primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    n = len(primes)
    
    # dp[i] is a list of tuples (length, last_gap, prev_idx, prev_state_idx)
    # representing non-dominated states ending at primes[i]
    # A state (L1, G1) dominates (L2, G2) if L1 >= L2 and G1 <= G2
    # We maintain dp[i] sorted by length descending, then gap ascending.
    # This allows us to quickly find the best (max length) valid state from a previous index.
    
    dp = [None] * n
    
    # Initialize
    dp[0] = [(1, 0, -1, -1)]
    best_len = 1
    best_i = 0
    best_state_idx = 0
    
    for i in range(1, n):
        if time.time() > deadline:
            break
            
        candidates = [(1, 0, -1, -1)] # Start new sequence at i
        
        for j in range(i):
            gap = primes[i] - primes[j]
            # Find the state in dp[j] with the maximum length L such that G <= gap.
            # dp[j] is sorted by L descending.
            # We iterate until we find the first state with G <= gap.
            # If found, it has the max L among valid states.
            best_state_from_j = None
            for k, (L, G, pi, si) in enumerate(dp[j]):
                if G <= gap:
                    best_state_from_j = (L + 1, gap, j, k)
                    break # Since sorted by L desc, this is the best one for this j
            
            if best_state_from_j:
                candidates.append(best_state_from_j)
        
        # Prune candidates to keep only non-dominated states
        # Sort by length descending, then by gap ascending
        candidates.sort(key=lambda x: (-x[0], x[1]))
        
        non_dom = []
        min_gap_encountered = float('inf')
        for cand in candidates:
            L, G, pi, si = cand
            if G < min_gap_encountered:
                non_dom.append(cand)
                min_gap_encountered = G
        
        dp[i] = non_dom
        
        # Update global best
        current_best_len = non_dom[0][0]
        if current_best_len > best_len:
            best_len = current_best_len
            best_i = i
            best_state_idx = 0

    # Reconstruct the subsequence
    seq = []
    i = best_i
    state_idx = best_state_idx
    while i != -1:
        L, G, prev_i, prev_state_idx = dp[i][state_idx]
        seq.append(primes[i])
        i = prev_i
        state_idx = prev_state_idx
        
    seq.reverse()
    return seq
