"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    N = 1000001  # we need lpf up to 1000001 (for n+1 when n=1000000)
    
    # Sieve of Eratosthenes to find all primes up to N
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    
    # Build largest prime factor table
    # lpf[m] = largest prime factor of m
    lpf = [0] * (N + 1)
    
    # For each prime p, set lpf[m] = p for all multiples m of p
    # Since we go in increasing order of p, the last assignment gives the largest prime factor
    for p in range(2, N + 1):
        if is_prime[p]:
            for m in range(p, N + 1, p):
                lpf[m] = p
    
    # Check time
    if time.time() > deadline - 5:
        # Not enough time, return a reasonable default
        return 10000
    
    # Find n in [10000, 1000000] that minimizes max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = max(lpf[10000], lpf[100001])
    
    for n in range(10001, 1000001):
        val = max(lpf[n], lpf[n+1])
        if val < best_val:
            best_val = val
            best_n = n
        if time.time() > deadline - 2:
            break
    
    return best_n
# EVOLVE-BLOCK-END
