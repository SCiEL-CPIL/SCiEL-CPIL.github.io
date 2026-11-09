import numpy as np
from math import isqrt

def construct():
    N = 10000
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = [i for i, val in enumerate(is_prime) if val]
    P = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    max_gap = N
    # Use int16 to save memory and potentially speed up
    dp = np.zeros((P, max_gap + 1), dtype=np.int16)
    prefix = np.zeros((P, max_gap + 1), dtype=np.int16)
    
    # Initialize
    for i in range(P):
        dp[i, 0] = 1
        prefix[i, 0] = 1
        prefix[i, 1:] = 1

    for j in range(1, P):
        p_j = primes[j]
        dp_row = dp[j]
        dp_row[0] = 1
        
        for i in range(j):
            g = p_j - primes[i]
            if g <= max_gap:
                val = 1 + prefix[i, g]
                if val > dp_row[g]:
                    dp_row[g] = val
        
        prefix[j] = np.maximum.accumulate(dp_row)
        
    lengths = prefix[:, max_gap]
    best_j = np.argmax(lengths)
    
    g_curr = np.argmax(dp[best_j])
    
    seq = []
    j_curr = best_j
    while g_curr > 0:
        p_curr = primes[j_curr]
        seq.append(p_curr)
        p_prev = p_curr - g_curr
        i_curr = prime_to_idx[p_prev]
        
        h = np.argmax(dp[i_curr, 0:g_curr+1])
        
        g_curr = h
        j_curr = i_curr
        
    seq.append(primes[j_curr])
    seq.reverse()
    
    return seq
