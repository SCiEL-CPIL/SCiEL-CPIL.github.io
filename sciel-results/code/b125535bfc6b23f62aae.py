import time
import math

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 90s. We aim to finish well before.
    # The exhaustive search over 10^6 numbers is very fast in Python (< 2s usually).
    deadline = start_time + 80
    
    N = 1000000
    max_val = N + 1  # We need pp up to N+1
    
    # Array to store powerful parts, 1-indexed
    # Using a list of integers is sufficient and fast for N=1e6
    pp = [1] * (max_val + 1)
    
    # Sieve of Eratosthenes to get primes up to sqrt(N+1)
    limit = int(math.isqrt(max_val))
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            # Mark multiples of i as not prime
            for j in range(i*i, limit+1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # For each prime, compute its contribution to powerful parts
    for p in primes:
        p2 = p * p
        if p2 > max_val:
            continue
        
        # Iterate through multiples of p^2
        # m = k * p^2
        # The exponent of p in m is 2 + (exponent of p in k)
        max_k = max_val // p2
        
        # Optimization: Precompute p^3, p^4... to check divisibility faster?
        # Or just use the while loop which is effectively counting trailing zeros in base p.
        
        for k in range(1, max_k + 1):
            m = k * p2
            
            # Find exact exponent of p in m
            # m is divisible by p^2.
            # Let temp = k. Count how many times p divides temp.
            temp = k
            extra_exp = 0
            while temp % p == 0:
                temp //= p
                extra_exp += 1
            
            total_exp = 2 + extra_exp
            # Multiply pp[m] by p^total_exp
            pp[m] *= p ** total_exp
        
        # Check deadline periodically
        if time.time() > deadline:
            break
    
    # Now find the max product pp[n] * pp[n+1] for n in [10000, 1000000]
    best_n = 10000
    max_prod = 0
    
    # Local variables for speed
    pp_local = pp
    for n in range(10000, N + 1):
        prod = pp_local[n] * pp_local[n + 1]
        if prod > max_prod:
            max_prod = prod
            best_n = n
        
        # Check deadline periodically
        # Checking time every iteration is expensive, so check every 10000
        if (n - 10000) % 10000 == 0 and time.time() > deadline:
            break
    
    return best_n
