"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    LIMIT = N_MAX + 1  # Need powerful_part for n and n+1, so up to 1000001
    
    # Linear sieve to compute powerful part for all integers up to LIMIT
    # pp[n] = product of p^e for all prime powers p^e (e>=2) dividing n
    
    # Using linear sieve approach:
    # For each number, we track:
    # - lp[i]: largest prime factor of i
    # - pe[i]: the highest power of lp[i] that divides i
    # - pp[i]: the powerful part of i
    
    lp = [0] * (LIMIT + 1)  # largest prime factor
    pe = [1] * (LIMIT + 1)  # highest power of lp[i] dividing i
    pp = [1] * (LIMIT + 1)  # powerful part
    
    primes = []
    
    for i in range(2, LIMIT + 1):
        if lp[i] == 0:
            lp[i] = i
            pe[i] = i
            primes.append(i)
        
        for p in primes:
            if p > lp[i] or i * p > LIMIT:
                break
            
            lp[i * p] = p
            
            if i % p == 0:
                # p divides i, so the power of p in i*p is pe[i] * p
                pe[i * p] = pe[i] * p
                # The powerful part: if pe[i] >= p*p (i.e., p^2 divides i), 
                # then p^e contributes to powerful part
                # pp[i*p] = pp[i] * (p^e if e >= 2 else 1)
                # But we need to be careful: pe[i] is the highest power of p in i
                # So in i*p, the highest power of p is pe[i]*p
                # If pe[i]*p >= p*p (i.e., pe[i] >= p), then p^(e+1) contributes
                # Actually, let me reconsider.
                
                # Let me use a different approach. For each number n,
                # pp[n] = product of p^e where p^e || n and e >= 2
                # 
                # In linear sieve, when we process i*p:
                # - If p | i: the p-part of i*p is pe[i]*p, and other prime parts same as i
                #   But we need to remove the old p-part from pp[i] and add the new one
                #   This is complex. Let me use a simpler approach.
                
                break
            else:
                # p does not divide i
                pe[i * p] = p
                # pp[i*p] = pp[i] * (p if p*p <= i*p else 1) but p*p > p since p>=2
                # Actually p^1 doesn't contribute (need e>=2), so pp[i*p] = pp[i]
                pass
        
        # This approach is getting complex. Let me use a cleaner method.
    
    # Let me restart with a cleaner linear sieve for powerful parts
    # 
    # Key insight: pp[n] = product of p^e for all primes p where p^e || n and e >= 2
    # 
    # In linear sieve:
    # - lp[i] = smallest prime factor of i (use smallest for cleaner logic)
    # - spf_power[i] = highest power of lp[i] that divides i
    # - pp[i] = powerful part
    #
    # When computing i = m * p where p = lp[i]:
    #   If p | m: spf_power[i] = spf_power[m] * p
    #             pp[i] = pp[m] / (spf_power[m] if spf_power[m] >= p*p else 1) * (spf_power[i] if spf_power[i] >= p*p else 1)
    #   If p ∤ m: spf_power[i] = p
    #             pp[i] = pp[m]  (since p^1 doesn't contribute)
    #
    # But division is expensive. Alternative:
    # Track separately the "contribution" from each prime.
    
    # Simpler approach: Use the original sieve but optimize it
    # Actually, let me just use a well-optimized version of the original approach
    
    # Sieve of Eratosthenes for primes up to sqrt(LIMIT)
    sqrt_limit = int(math.isqrt(LIMIT))
    is_prime = [True] * (sqrt_limit + 1)
    if sqrt_limit >= 0:
        is_prime[0] = False
    if sqrt_limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(sqrt_limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, sqrt_limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Initialize powerful part array
    pp = [1] * (LIMIT + 1)
    
    # For each prime, process its powers
    for p in primes:
        p2 = p * p
        if p2 > LIMIT:
            break
        
        # For each multiple of p^2, determine the exact power of p
        # and multiply pp by that power
        for m in range(p2, LIMIT + 1, p2):
            # m is divisible by p^2
            # Find the exact exponent e of p in m
            temp = m
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            # e >= 2 guaranteed since p^2 | m
            pp[m] *= (p ** e)
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # If we ran out of time during sieve, we may have incomplete pp array
    # But let's proceed with what we have
    
    # Scan for the best n in [N_MIN, N_MAX] maximizing pp[n] * pp[n+1]
    best_n = N_MIN
    best_product = pp[N_MIN] * pp[N_MIN + 1]
    
    for n in range(N_MIN, N_MAX + 1):
        prod = pp[n] * pp[n + 1]
        if prod > best_product:
            best_product = prod
            best_n = n
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
