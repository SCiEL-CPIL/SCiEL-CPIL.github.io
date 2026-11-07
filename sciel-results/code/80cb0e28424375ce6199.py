# EVOLVE-BLOCK-START
import math
import time

def construct():
    deadline = time.time() + 85  # 85s safety margin before 90s kill
    N = 256
    
    # Sieve of Eratosthenes to find all primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(math.isqrt(N)) + 1):
        if is_prime[p]:
            for m in range(p * p, N + 1, p):
                is_prime[m] = False
    primes = [p for p in range(2, N + 1) if is_prime[p]]
    
    # For each prime, find the largest power of that prime <= N
    selected = [1]  # include 1
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        selected.append(power)
    
    # Verify pairwise coprime (sanity check)
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            if math.gcd(selected[i], selected[j]) != 1:
                # Fallback: just return primes + 1
                return [1] + primes
    
    # Ensure all are Python ints
    result = [int(x) for x in selected]
    return result
# EVOLVE-BLOCK-END
