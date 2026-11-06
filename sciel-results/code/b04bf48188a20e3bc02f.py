import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 1000000
    # We need P(n) for n in [10000, 1000000] and P(n+1) for n in [10000, 999999]
    # So we need P(k) for k in [10000, 1000000]
    
    # Generate primes up to sqrt(1000000) = 1000
    limit = int(math.isqrt(N)) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # Precompute powerful parts
    # We only need indices 10000 to 1000000
    # But let's compute for a slightly wider range if needed, or just the specific range.
    # Computing for all 1..N is easier to code but wastes time on 1..9999.
    # 1..9999 is 1% of the range. Negligible.
    
    P = [1] * (N + 2)
    
    for n in range(10000, N + 1):
        t = n
        pn = 1
        for p in primes:
            if p * p > t:
                break
            if t % p == 0:
                cnt = 0
                while t % p == 0:
                    t //= p
                    cnt += 1
                if cnt >= 2:
                    pn *= p ** cnt
        # If t > 1, it is a prime factor with exponent 1, so it doesn't contribute to P(n)
        # No need to do anything with t
        
        P[n] = pn
        
        # Check time
        if n % 100000 == 0 and time.time() > deadline:
            # If we are running out of time, we might not finish precomputation.
            # But the loop is fast. If we are here, we are stuck?
            # Actually, if we haven't finished precomputation, we can't find the best n.
            # We should probably just return a good candidate found so far?
            # But we haven't scored any pairs yet.
            # Let's assume it finishes.
            pass

    best_n = 10000
    best_val = 0
    
    for n in range(10000, N):
        val = P[n] * P[n+1]
        if val > best_val:
            best_val = val
            best_n = n
            
        if n % 100000 == 0 and time.time() > deadline:
            break

    return best_n
