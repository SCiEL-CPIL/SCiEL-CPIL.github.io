import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # 5s safety margin before 90s kill
    
    N = 1000000
    MAX_N = N + 1  # We need P[N+1]
    
    # Sieve of Eratosthenes for primes up to sqrt(MAX_N)
    limit = int(math.isqrt(MAX_N))
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Initialize powerful part array
    P = [1] * (MAX_N + 1)
    
    # Fill P
    for p in primes:
        q = p * p
        # q is p^k, starting with k=2
        while q <= MAX_N:
            # For all m = q * i such that i is not divisible by p
            # m <= MAX_N => i <= MAX_N // q
            max_i = MAX_N // q
            
            # Optimized loop: skip multiples of p
            # i ranges from 1 to max_i
            # We want i % p != 0
            # Iterate in blocks of p
            full_blocks = max_i // p
            remainder = max_i % p
            
            for j in range(full_blocks + 1):
                start_i = j * p + 1
                end_i = start_i + p - 1
                if end_i > max_i:
                    end_i = max_i
                for i in range(start_i, end_i + 1):
                    P[q * i] *= q
            
            q *= p
            
            # Check deadline
            if time.time() - start_time > DEADLINE:
                break
        
        if time.time() - start_time > DEADLINE:
            break
    
    # Find max product
    best_n = 10000
    best_val = 0
    # n in [10000, 1000000]
    # We need P[n] and P[n+1]
    for n in range(10000, N + 1):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
            
        if time.time() - start_time > DEADLINE:
            break
            
    return best_n
