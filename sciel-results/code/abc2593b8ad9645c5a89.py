import numpy as np
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 10000
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
                
    primes = [i for i, v in enumerate(is_prime) if v]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    max_gap = N
    
    # Precompute transitions
    # transitions[i] is a list of (g, j)
    transitions = [[] for _ in range(num_primes)]
    for i in range(num_primes):
        p_i = primes[i]
        # g can be up to p_i - 2 (since min prime is 2)
        # Actually g can be 1 (if p_i=3, prev=2)
        for g in range(1, p_i - 1):
            prev_val = p_i - g
            if is_prime[prev_val]:
                j = prime_to_idx[prev_val]
                transitions[i].append((g, j))
                
    # DP Arrays
    dp = np.zeros((num_primes, max_gap + 1), dtype=np.uint16)
    pref_max = np.zeros((num_primes, max_gap + 1), dtype=np.uint16)
    pref_argmax = np.zeros((num_primes, max_gap + 1), dtype=np.int32)
    
    prev_idx = np.full((num_primes, max_gap + 1), -1, dtype=np.int32)
    prev_gap = np.full((num_primes, max_gap + 1), -1, dtype=np.int32)
    
    best_len = 1
    best_i = 0
    best_g = 0
    
    for i in range(num_primes):
        # Base case: sequence of length 1, gap 0
        dp[i, 0] = 1
        pref_max[i, 0] = 1
        pref_argmax[i, 0] = 0
        
        # Process transitions
        for g, j in transitions[i]:
            # j < i is guaranteed
            best_prev = pref_max[j, g]
            if best_prev > 0:
                dp[i, g] = best_prev + 1
                prev_idx[i, g] = j
                prev_gap[i, g] = pref_argmax[j, g]
                if dp[i, g] > best_len:
                    best_len = dp[i, g]
                    best_i = i
                    best_g = g
        
        # Compute prefix max for i
        # We only need to compute up to max g in transitions? 
        # No, pref_max is used for future i's with any g.
        # So we must compute for all g up to max_gap.
        # But dp[i, g] is 0 for most g.
        # We can just loop 1 to max_gap.
        # 10000 iterations * 1229 = 12 million. Fast enough.
        
        curr_max = 1
        curr_argmax = 0
        for g in range(1, max_gap + 1):
            if dp[i, g] > curr_max:
                curr_max = dp[i, g]
                curr_argmax = g
            pref_max[i, g] = curr_max
            pref_argmax[i, g] = curr_argmax
            
        # Check deadline
        if time.time() > deadline:
            break
            
    # Reconstruct
    if best_len <= 1:
        return [2] # or [primes[0]]
        
    path = []
    i = best_i
    g = best_g
    while i >= 0 and g >= 0:
        path.append(primes[i])
        if g == 0:
            break
        next_i = prev_idx[i, g]
        next_g = prev_gap[i, g]
        i = next_i
        g = next_g
        
    path.reverse()
    return path
