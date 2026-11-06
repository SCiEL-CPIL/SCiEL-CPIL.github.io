import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin, evaluator kills at 90s. 35s buffer for safety.
    
    N_MAX = 1000000
    N_LIMIT = N_MAX + 1  # We need n+1 up to 1000001
    
    # 1. Sieve for smallest prime factor (SPF)
    spf = list(range(N_LIMIT + 1))
    for i in range(2, int(N_LIMIT**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N_LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Precompute powerful parts using SPF and DP
    # pp[n] = powerful part of n
    # rem[n] = n with all factors of spf[n] removed
    # exp[n] = exponent of spf[n] in n
    pp = [1] * (N_LIMIT + 1)
    rem = [0] * (N_LIMIT + 1)
    exp = [0] * (N_LIMIT + 1)
    
    pp[1] = 1
    rem[1] = 1
    exp[1] = 0
    
    for n in range(2, N_LIMIT + 1):
        p = spf[n]
        n_prime = n // p
        
        if n_prime % p == 0:
            exp[n] = exp[n_prime] + 1
            rem[n] = rem[n_prime]
        else:
            exp[n] = 1
            rem[n] = n_prime
            
        if exp[n] >= 2:
            pp[n] = pp[rem[n]] * (p ** exp[n])
        else:
            pp[n] = pp[rem[n]]
            
        if time.time() > deadline:
            # Should not happen, but safe fallback
            break
    
    # 3. Find best n via Brute Force
    best_score = -1
    best_n = 10000
    
    # Range [10000, 1000000]
    for n in range(10000, N_MAX + 1):
        score = pp[n] * pp[n + 1]
        if score > best_score:
            best_score = score
            best_n = n
            
    # 4. CRT-based Construction Search
    # Generate prime powers
    prime_powers = set()
    for p in range(2, N_LIMIT + 1):
        if spf[p] == p: # p is prime
            power = p
            while power <= N_LIMIT:
                prime_powers.add(power)
                power *= p
                
    # Sort descending to prioritize large products
    pp_list = sorted([x for x in prime_powers if x <= N_MAX], reverse=True)
    
    # Optimization: Only consider A, B such that A*B > best_score
    # Since pp_list is sorted descending, if A * B <= best_score, 
    # then for any j' > j, A * pp_list[j'] <= A * B <= best_score.
    # So we can break the inner loop.
    
    for i in range(len(pp_list)):
        A = pp_list[i]
        
        # If A * largest_B (which is pp_list[i+1] roughly) is small, break outer?
        # No, just inner.
        
        for j in range(i + 1, len(pp_list)):
            B = pp_list[j]
            prod = A * B
            
            # Since B is decreasing, if prod <= best_score, no further B will work
            if prod <= best_score:
                break
                
            # Solve n = A*k, n+1 = B*m
            # A*k - B*m = -1
            # k*A = 1 mod B
            # k = A^{-1} mod B
            try:
                k_inv = pow(A, -1, B)
            except ValueError:
                continue # Should be coprime
                
            k = k_inv
            n0 = A * k
            step = A * B
            
            # Find t such that 10000 <= n0 + t*step <= 1000000
            if n0 < 10000:
                t_min = (10000 - n0 + step - 1) // step
            else:
                t_min = 0
                
            t_max = (1000000 - n0) // step
            
            if t_min <= t_max:
                # Usually only one or few solutions in range
                for t in range(t_min, t_max + 1):
                    n_cand = n0 + t * step
                    if 10000 <= n_cand <= N_MAX:
                        # Verify score
                        # pp[n_cand] should be divisible by A
                        # pp[n_cand+1] should be divisible by B
                        score_cand = pp[n_cand] * pp[n_cand + 1]
                        if score_cand > best_score:
                            best_score = score_cand
                            best_n = n_cand
                            
        if time.time() > deadline:
            break
            
    return best_n
