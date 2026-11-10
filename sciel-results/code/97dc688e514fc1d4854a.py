# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    LIMIT = MAX_N + 1  # We need pp[MAX_N+1]
    
    # Build smallest prime factor sieve using linear sieve for speed
    spf = list(range(LIMIT + 1))
    primes = []
    for i in range(2, LIMIT + 1):
        if spf[i] == i:
            primes.append(i)
        for p in primes:
            if p * i > LIMIT or p > spf[i]:
                break
            spf[p * i] = p
    
    # Compute powerful part for each number in [1, LIMIT]
    # pp[n] = product of p^e for each prime factor p with exponent e >= 2
    pp = [1] * (LIMIT + 1)
    
    # For each prime p, find all numbers divisible by p^2 and add p^2 * (p^(e-2))
    # More efficient: for each prime p, iterate through powers p^2, p^3, ...
    # and for each number divisible by p^k but not p^(k+1), multiply pp by p^k
    
    for p in primes:
        if p * p > LIMIT:
            break
        
        power = p * p
        while power <= LIMIT:
            # For each multiple of 'power'
            # Check if it's also divisible by next_power
            next_power = power * p
            for m in range(power, LIMIT + 1, power):
                if m % next_power != 0:
                    pp[m] *= power
            
            power = next_power
    
    # Find n in [MIN_N, MAX_N] that maximizes pp[n] * pp[n+1]
    best_n = MIN_N
    best_val = pp[MIN_N] * pp[MIN_N + 1]
    
    for n in range(MIN_N, MAX_N + 1):
        v = pp[n] * pp[n + 1]
        if v > best_val:
            best_val = v
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
