"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    MAX_VAL = N_MAX + 1  # We need LPF up to 1000001
    
    # Precompute Largest Prime Factor for all numbers up to MAX_VAL
    # Using a sieve that iterates primes in increasing order.
    # lpf[i] will hold the largest prime factor of i.
    lpf = [0] * (MAX_VAL + 1)
    
    # Sieve of Eratosthenes variant for Largest Prime Factor
    # Iterate i from 2 to MAX_VAL
    # If lpf[i] is 0, i is prime.
    # Mark all multiples of i with i.
    # Since i increases, the last mark is the largest prime factor.
    
    # Optimization: We only need to iterate i up to MAX_VAL.
    # However, standard sieve logic:
    for i in range(2, MAX_VAL + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, MAX_VAL + 1, i):
                lpf[j] = i
                
    # The sieve above is O(N log log N).
    # For N=10^6, this is fast.
    
    # Scan the range [N_MIN, N_MAX] to find n minimizing max(lpf[n], lpf[n+1])
    best_n = N_MIN
    # Initialize best_val with the first candidate
    best_val = max(lpf[N_MIN], lpf[N_MIN + 1])
    
    # We can optimize the scan by checking time periodically
    # But the scan is 1M iterations, which is very fast in Python (< 0.1s)
    # The sieve is the main cost.
    
    for n in range(N_MIN + 1, N_MAX + 1):
        # Time guard
        if time.time() > deadline:
            break
            
        # Calculate largest prime factor of n(n+1)
        # Since gcd(n, n+1) = 1, lpf(n(n+1)) = max(lpf(n), lpf(n+1))
        current_val = lpf[n]
        next_val = lpf[n + 1]
        if next_val > current_val:
            current_val = next_val
            
        if current_val < best_val:
            best_val = current_val
            best_n = n
            
    return best_n

# EVOLVE-BLOCK-END
