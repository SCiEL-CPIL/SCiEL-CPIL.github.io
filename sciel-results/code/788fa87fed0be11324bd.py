import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # 45s safety margin for 60s limit
    
    N = 123457
    half_N = N // 2
    
    # Precompute primes up to sqrt(N)
    limit = int(math.isqrt(N)) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    def largest_prime_factor(n):
        if n == 1:
            return 1
        lpf = 1
        temp_n = n
        for p in primes:
            if p * p > temp_n:
                break
            if temp_n % p == 0:
                lpf = p
                while temp_n % p == 0:
                    temp_n //= p
        if temp_n > 1:
            lpf = max(lpf, temp_n)
        return lpf

    best_score = float('inf')
    best_a = 1
    best_b = N - 1

    for a in range(1, half_N + 1):
        # Time check
        if time.time() > deadline:
            break
            
        b = N - a
        pa = largest_prime_factor(a)
        pb = largest_prime_factor(b)
        current_max = max(pa, pb)
        
        if current_max < best_score:
            best_score = current_max
            best_a = a
            best_b = b
            
            # If we find a theoretical lower bound, we can stop.
            # Since N is odd, one is even, one is odd.
            # If a=1, b=N. P(1)=1. P(N) is likely large.
            # The absolute minimum for max(P(a), P(b)) where a+b is odd:
            # If a=1, P(a)=1. P(b) is fixed.
            # If a>1, b>1, one is odd >1, so P(odd) >= 3.
            # So min possible is 3.
            if best_score == 3:
                break

    return [best_a, best_b]
