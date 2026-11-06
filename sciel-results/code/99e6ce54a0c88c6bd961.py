import numpy as np
import time
from math import isqrt

def construct():
    start_time = time.time()
    
    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.array([i for i, v in enumerate(is_prime) if v], dtype=np.int32)
    N = len(primes)
    
    # 2. Initialize DP arrays
    # Max gap is limit
    max_gap = limit
    # dp[i][g] = max length of sequence ending at primes[i] with last gap exactly g
    dp = np.zeros((N, max_gap + 1), dtype=np.int16)
    # pref[i][g] = max length of sequence ending at primes[i] with last gap <= g
    pref = np.zeros((N, max_gap + 1), dtype=np.int16)
    # parent[i][g] = index j of the previous prime in the optimal sequence
    parent = np.full((N, max_gap + 1), -1, dtype=np.int32)
    
    # Base case: A sequence of length 1 ending at any prime.
    # We model this as having a "gap" of 0.
    dp[:, 0] = 1
    pref[:, 0] = 1
    # For g > 0, pref should also be at least 1 initially?
    # If we start at j and take gap g, length is 2.
    # pref[j][g] needs to be 1 to allow this.
    # np.maximum.accumulate will handle this if dp[:,0]=1.
    # So we just need to compute pref for each i after filling dp[i].
    
    best_len = 1
    best_i = 0
    best_g = 0
    
    for i in range(N):
        if time.time() - start_time > 50:
            break
            
        pi = primes[i]
        
        # Update dp[i] based on all previous j
        for j in range(i):
            g = pi - primes[j]
            if g > max_gap:
                continue
            # We can extend any sequence ending at j with gap <= g
            # The best length is pref[j][g]
            new_len = pref[j][g] + 1
            if new_len > dp[i][g]:
                dp[i][g] = new_len
                parent[i][g] = j
        
        # Compute prefix max for dp[i] to get pref[i]
        # pref[i][g] = max(dp[i][0], dp[i][1], ..., dp[i][g])
        pref[i] = np.maximum.accumulate(dp[i])
        
        # Update global best
        # The max length ending at i is pref[i][max_gap]
        current_max = pref[i][max_gap]
        if current_max > best_len:
            best_len = current_max
            # Find the gap g that achieves this max
            # We want the largest g? Or any g?
            # For reconstruction, we just need one valid g.
            # Let's pick the first g from the right (largest gap) or left?
            # It doesn't matter for validity.
            # np.where(dp[i] == current_max)[0]
            # Let's pick the largest gap to be deterministic or just first.
            indices = np.where(dp[i] == current_max)[0]
            if len(indices) > 0:
                best_g = indices[-1] # Pick largest gap
                best_i = i
            else:
                # Should not happen if current_max > 0
                best_i = i
                best_g = 0

    # Reconstruct sequence
    if best_len <= 1:
        return [primes[0].item()]
        
    seq = []
    i = best_i
    g = best_g
    
    while i != -1:
        seq.append(primes[i].item())
        if g == 0:
            break
        
        j = parent[i][g]
        if j == -1:
            break
            
        # Find the gap g' at j that was used
        # It must be <= g and dp[j][g'] == pref[j][g]
        # We scan from g down to 0
        target_len = pref[j][g]
        found = False
        for g_prime in range(g, -1, -1):
            if dp[j][g_prime] == target_len:
                g = g_prime
                i = j
                found = True
                break
        
        if not found:
            # Fallback, should not happen
            break

    seq.reverse()
    return seq
