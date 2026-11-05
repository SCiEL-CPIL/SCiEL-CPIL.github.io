import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s
    
    N = 1000000
    # powerful_part[i] will store the powerful part of i
    powerful_part = [1] * (N + 2)  # +2 to safely access n+1 when n=N
    
    # Sieve to compute powerful parts
    # We need to identify primes and their powers
    # A simple sieve of Eratosthenes to get primes up to sqrt(N) is not enough,
    # we need primes up to N/2? No, just iterate i from 2 to N.
    # If i is prime, process its powers.
    
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    for p in range(2, N + 1):
        if time.time() > deadline:
            break
        if is_prime[p]:
            # Process powers of p
            power = p * p
            k = 2
            while power <= N:
                # For each multiple m of 'power', check if the exponent of p in m is exactly k
                # i.e., m is divisible by p^k but not p^(k+1)
                next_power = power * p
                for m in range(power, N + 1, power):
                    if m % next_power != 0:
                        powerful_part[m] *= power
                power = next_power
                k += 1
                
    best_n = 10000
    best_score = -1
    
    # Find the n that maximizes P(n) * P(n+1)
    for n in range(10000, N):
        if time.time() > deadline:
            break
        score = powerful_part[n] * powerful_part[n + 1]
        if score > best_score:
            best_score = score
            best_n = n
            
    return best_n
