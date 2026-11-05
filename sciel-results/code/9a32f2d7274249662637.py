import time
import math

def construct():
    deadline = time.time() + 80  # Safety margin before 90s limit
    N = 1000000
    
    # Sieve of Eratosthenes to get primes up to N
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Compute powerful part P(n) for all n in [1, N]
    P = [1] * (N + 1)
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        # For each multiple of p^2, find the full exponent of p
        for m in range(p2, N + 1, p2):
            # Count exponent of p in m
            e = 2
            temp = m
            while temp % (p2 * p) == 0:  # temp % p^3 == 0 means e >= 3
                e += 1
                temp //= p
            # Actually, let me recount properly
            e = 0
            temp = m
            while temp % p == 0:
                e += 1
                temp //= p
            # e >= 2 is guaranteed since m is multiple of p^2
            P[m] *= (p ** e)
    
    # Find max P[n] * P[n+1] for n in [10000, N-1]
    best_val = 0
    best_n = 10000
    for n in range(10000, N):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        if time.time() > deadline:
            break
    
    return best_n
