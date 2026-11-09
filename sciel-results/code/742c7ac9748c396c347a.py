"""Longest increasing prime subsequence with nondecreasing gaps, <=10000."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 60s (or 90s) kill

    # Step 1: Generate all primes <= 10000 using sieve
    limit = 10000
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    primes = np.where(sieve)[0]
    n = len(primes)
    
    # Step 2: DP for longest subsequence with nondecreasing gaps
    # dp[i] = length of longest valid subsequence ending at prime index i
    # last_gap[i] = the gap of the last step in that subsequence (0 if length 1)
    # prev[i] = previous index in the subsequence (-1 if none)
    
    dp = np.ones(n, dtype=np.int32)
    last_gap = np.zeros(n, dtype=np.int32)  # gap before primes[i] in the subsequence
    prev = np.full(n, -1, dtype=np.int32)
    
    # For each prime i, try to extend subsequences ending at j < i
    # where primes[i] - primes[j] >= last_gap[j]
    # We want to maximize dp[j] + 1
    
    # Vectorized approach: for each i, find all j < i such that gap >= last_gap[j]
    # Among those, pick the one with max dp[j]
    
    # This is O(n^2) but with vectorized inner loop
    best_len = 1
    best_end = 0
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        
        # Gaps from each j to i
        gaps = primes[i] - primes[:i]
        
        # Valid transitions: gap >= last_gap[j]
        valid = gaps >= last_gap[:i]
        
        if not np.any(valid):
            continue
        
        # Among valid j, find the one with maximum dp[j]
        # We want max dp[j] for valid j
        valid_dp = np.where(valid, dp[:i], 0)
        max_idx = np.argmax(valid_dp)
        
        if valid_dp[max_idx] > 0:
            candidate_len = valid_dp[max_idx] + 1
            if candidate_len > dp[i]:
                dp[i] = candidate_len
                last_gap[i] = gaps[max_idx]
                prev[i] = max_idx
        
        if dp[i] > best_len:
            best_len = dp[i]
            best_end = i
    
    # Step 3: Reconstruct the sequence
    seq = []
    idx = best_end
    while idx >= 0:
        seq.append(int(primes[idx]))
        idx = int(prev[idx])
    seq.reverse()
    
    return seq
# EVOLVE-BLOCK-END
