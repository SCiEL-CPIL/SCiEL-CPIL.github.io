"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    N = 1000000
    MIN_N = 10000
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0 # 10 seconds margin before 90s

    # 1. Sieve for Primes
    is_prime = [True] * (N + 2)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            if i * i <= N:
                for j in range(i * i, N + 1, i):
                    is_prime[j] = False

    # 2. Compute powerful part for all numbers up to N
    # pp[n] will hold the powerful part of n
    pp = [1] * (N + 2)
    
    # For each prime p, for each power p^e with e >= 2:
    #   For each multiple m of p^e that is NOT a multiple of p^(e+1):
    #     pp[m] *= p^e
    # This correctly assigns the full p^e factor only when the exact exponent is e.
    
    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        
        # Iterate over powers: p^2, p^3, p^4, ...
        power = p2
        e = 2
        while power <= N:
            next_power = power * p
            # For multiples of power that are not multiples of next_power
            # (i.e., exact exponent is e)
            for m in range(power, N + 1, power):
                if m % next_power != 0:
                    pp[m] *= power
            
            power = next_power
            e += 1
            
            # Check deadline
            if time.time() > deadline:
                return MIN_N # Fallback, though unlikely to happen this fast

    # 3. Find max product
    best_n = MIN_N
    best_val = 0

    for n in range(MIN_N, N):
        # Check deadline
        if time.time() > deadline:
            break
        
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n

    return best_n

# EVOLVE-BLOCK-END
