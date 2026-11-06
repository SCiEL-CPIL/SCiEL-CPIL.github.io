import numpy as np
from math import isqrt

def construct():
    # Generate primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    N = len(primes)
    
    # DP arrays
    # dp[i][g]: max length of sequence ending at primes[i] with last gap exactly g
    # best_len[i][g]: max length of sequence ending at primes[i] with last gap <= g
    # best_gap[i][g]: the gap h <= g that achieves best_len[i][g]
    # parent_j[i][g]: index j of previous prime in the optimal path for state (i, g)
    # parent_h[i][g]: gap h used at prime j in the optimal path for state (i, g)
    
    # Initialize
    dp = np.zeros((N, limit + 1), dtype=np.int16)
    best_len = np.zeros((N, limit + 1), dtype=np.int16)
    best_gap = np.zeros((N, limit + 1), dtype=np.int32)
    parent_j = np.full((N, limit + 1), -1, dtype=np.int16)
    parent_h = np.zeros((N, limit + 1), dtype=np.int32)
    
    # Fill DP
    for i in range(N):
        # Base case: sequence of length 1 starting at primes[i]
        # We model this as having a "previous gap" of 0
        dp[i, 0] = 1
        
        # Transition from all j < i
        for j in range(i):
            g = primes[i] - primes[j]
            # If g is 0, skip (primes are distinct)
            # best_len[j, g] is the max length ending at j with gap <= g
            # We can extend this sequence by adding primes[i]
            # The new gap is g
            # The length becomes best_len[j, g] + 1
            # We update dp[i, g]
            candidate_len = best_len[j, g] + 1
            if candidate_len > dp[i, g]:
                dp[i, g] = candidate_len
                # Store backpointers
                # The state at j that gave best_len[j, g] had a specific gap h
                # which is best_gap[j, g]
                parent_j[i, g] = j
                parent_h[i, g] = best_gap[j, g]
        
        # Compute prefix maximums for best_len and best_gap
        # best_len[i, g] = max(dp[i, 0], ..., dp[i, g])
        # We can use cummax, but we also need to track the argmax (best_gap)
        # cummax doesn't directly give argmax in a way that's easy to vectorize for tracking
        # So we do a simple loop for each i, which is O(limit)
        # N * limit = 1229 * 10000 = 12M operations. This is acceptable in Python if optimized, 
        # but a Python loop of 12M might take ~1s. 
        # Let's try to vectorize or optimize.
        
        # Simple loop is probably fine given the 90s budget.
        max_l = 0
        max_g = 0
        for g in range(limit + 1):
            if dp[i, g] > max_l:
                max_l = dp[i, g]
                max_g = g
            best_len[i, g] = max_l
            best_gap[i, g] = max_g
            
    # Find the global maximum
    max_total_len = 0
    best_end_i = -1
    best_end_g = -1
    for i in range(N):
        if best_len[i, limit] > max_total_len:
            max_total_len = best_len[i, limit]
            best_end_i = i
            best_end_g = limit
            
    # Backtrack to reconstruct the sequence
    sequence = []
    cur_i = best_end_i
    cur_g = best_end_g
    while cur_i != -1:
        sequence.append(primes[cur_i])
        # Get the previous state
        prev_j = parent_j[cur_i, cur_g]
        prev_h = parent_h[cur_i, cur_g]
        cur_i = prev_j
        cur_g = prev_h
        
    # The sequence is built in reverse
    sequence.reverse()
    return sequence

# EVOLVE-BLOCK-END
