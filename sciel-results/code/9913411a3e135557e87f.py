import numpy as np
from math import isqrt

def construct():
    # Generate primes <= 10000
    N = 10000
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = [i for i, val in enumerate(is_prime) if val]
    P = len(primes)
    
    # Map prime value to index
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    max_gap = N
    # dp[i][g]: max length ending at primes[i] with last gap g
    dp = np.zeros((P, max_gap + 1), dtype=np.int32)
    # prefix[i][g]: max length ending at primes[i] with last gap <= g
    prefix = np.zeros((P, max_gap + 1), dtype=np.int32)
    # argmax[i][g]: the gap g' <= g that achieves prefix[i][g]
    argmax = np.zeros((P, max_gap + 1), dtype=np.int32)
    
    # Initialize: sequence of length 1 has gap 0
    for i in range(P):
        dp[i, 0] = 1
        prefix[i, 0] = 1
        argmax[i, 0] = 0
        
    # Process primes in increasing order
    for j in range(1, P):
        p_j = primes[j]
        # Initialize dp[j, 0] = 1 is already done
        # For all previous primes i < j
        for i in range(j):
            p_i = primes[i]
            g = p_j - p_i
            # g is the gap from p_i to p_j
            # We can extend a sequence ending at p_i with last gap <= g
            if g <= max_gap:
                candidate = 1 + prefix[i, g]
                if candidate > dp[j, g]:
                    dp[j, g] = candidate
                    
        # Compute prefix and argmax for j
        # prefix[j, g] = max(prefix[j, g-1], dp[j, g])
        # We can do this with a loop or vectorized ops
        # Loop is fine for 10000
        for g in range(1, max_gap + 1):
            if dp[j, g] > prefix[j, g-1]:
                prefix[j, g] = dp[j, g]
                argmax[j, g] = g
            else:
                prefix[j, g] = prefix[j, g-1]
                argmax[j, g] = argmax[j, g-1]
                
    # Find the best ending
    best_len = 0
    best_idx = -1
    best_gap = -1
    for i in range(P):
        # The max length ending at i is prefix[i, max_gap]
        l = prefix[i, max_gap]
        if l > best_len:
            best_len = l
            best_idx = i
            # We need the gap that achieves this max
            # It's argmax[i, max_gap]
            best_gap = argmax[i, max_gap]
            
    # Reconstruct sequence
    seq = []
    idx = best_idx
    g = best_gap
    while g > 0:
        p = primes[idx]
        seq.append(p)
        # Find previous prime index
        prev_p = p - g
        prev_idx = prime_to_idx[prev_p]
        # Find previous gap
        g = argmax[prev_idx, g]
        idx = prev_idx
        
    # Add the first element (when g becomes 0)
    seq.append(primes[idx])
    
    seq.reverse()
    return seq
