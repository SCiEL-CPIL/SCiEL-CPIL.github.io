"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # Build smallest prime factor sieve up to MAX_N + 1
    spf = list(range(MAX_N + 2))
    
    # Sieve of Eratosthenes
    limit = int(math.isqrt(MAX_N + 1))
    for i in range(2, limit + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 2, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number using a sieve-based approach
    # For each prime p, for each power p^e with e>=2, multiply all multiples of p^e by p
    # This is equivalent to: pp[n] = product of p^e for each prime p where p^e || n and e>=2
    
    pp = [1] * (MAX_N + 2)
    
    # For each prime p, compute all powers p^2, p^3, ... up to MAX_N
    # For each such power p^e (e>=2), for each multiple m of p^e, multiply pp[m] by p
    # Wait, that's not quite right. If n = p^e * k where gcd(k,p)=1 and e>=2,
    # then the powerful part includes p^e.
    # 
    # Better approach: for each number, factor it using spf.
    # But let's try a different sieve approach:
    # 
    # For each prime p:
    #   For e = 2, 3, 4, ... while p^e <= MAX_N:
    #     For each multiple m of p^e:
    #       Check if the exact power of p in m is at least e
    #       If so, we need to include p^e in pp[m]
    # 
    # Actually, the simplest correct approach is still per-number factorization with spf.
    # Let's optimize that loop.
    
    # Precompute p^e values for quick reference
    # Actually, let's just optimize the inner loop
    
    for n in range(MIN_N, MAX_N + 2):
        x = n
        val = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                # Multiply by p^e
                for _ in range(e):
                    val *= p
        pp[n] = val
        
        # Check time every 50000 iterations to avoid overhead
        if n % 50000 == 0:
            if time.time() > deadline - 2:
                break
    
    # Find best n
    best_n = MIN_N
    best_val = 0
    
    # Determine how far we computed pp
    last_computed = MAX_N + 1
    if time.time() > deadline - 2:
        # We may have broken early; find last computed
        # Actually, we'll just iterate up to MIN_N + (MAX_N - MIN_N) but check pp[n+1]
        pass
    
    for n in range(MIN_N, MAX_N):
        prod = pp[n] * pp[n + 1]
        if prod > best_val:
            best_val = prod
            best_n = n
            
    return best_n
# EVOLVE-BLOCK-END
