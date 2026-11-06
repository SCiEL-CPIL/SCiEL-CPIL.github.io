import time
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin from 90s limit
    
    # 1. Generate primes up to 10000
    limit_val = 10000
    is_prime = bytearray(b'\x01') * (limit_val + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit_val)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_val + 1, i):
                is_prime[j] = 0
    
    primes = [i for i, flag in enumerate(is_prime) if flag]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes
    
    # 2. DP Setup
    # dp[e][p] = length of longest valid subsequence ending at primes[e] with previous prime primes[p]
    # choice[e][p] = index k such that dp[p][k] was the max used to compute dp[e][p]
    dp = [[0] * N for _ in range(N)]
    choice = [[-1] * N for _ in range(N)]
    
    # For each p, we maintain a pointer to the next k to consider
    # and the current max value for the range [k_ptr+1, p-1]
    k_ptr = [p - 1 for p in range(N)]
    max_val = [0] * N
    argmax_idx = [-1] * N
    
    best_len = 2
    best_e = 1
    best_p = 0
    
    # 3. DP Iteration
    # Precompute prime values as a list for faster access
    pvals = primes  # alias for clarity
    
    for e in range(N):
        if time.time() - start_time > deadline:
            break
        pe = pvals[e]
        for p in range(e):
            pp = pvals[p]
            gap = pe - pp
            limit = pp - gap  # = 2*pp - pe
            
            # Update max_val[p] with new k's that become valid
            # k must satisfy primes[k] >= limit
            kp = k_ptr[p]
            mv = max_val[p]
            ai = argmax_idx[p]
            
            while kp >= 0 and pvals[kp] >= limit:
                k = kp
                dpk = dp[p][k]
                if dpk > mv:
                    mv = dpk
                    ai = k
                kp -= 1
            
            k_ptr[p] = kp
            max_val[p] = mv
            argmax_idx[p] = ai
            
            if mv > 0:
                dpe = mv + 1
                choice[e][p] = ai
            else:
                dpe = 2
                choice[e][p] = -1
            
            dp[e][p] = dpe
            
            if dpe > best_len:
                best_len = dpe
                best_e = e
                best_p = p
                
    # 4. Reconstruct Sequence
    seq = []
    if best_len >= 2:
        seq.append(primes[best_e])
        seq.append(primes[best_p])
        curr_e, curr_p = best_e, best_p
        k = choice[curr_e][curr_p]
        while k != -1:
            seq.append(primes[k])
            curr_e, curr_p = curr_p, k
            k = choice[curr_e][curr_p]
        
    seq.reverse()
    return seq
