import time
import numpy as np

def construct():
    start = time.time()
    DEADLINE = 60.0  # safety margin before the 90s kill

    N = 1000000          # max allowed n
    NMAX = N + 1         # we need lpf for values up to 1000001

    # --- Sieve primes up to NMAX ---
    # Use boolean array for sieve
    is_prime = np.ones(NMAX + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    # Standard Sieve of Eratosthenes
    for i in range(2, int(NMAX**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i : NMAX+1 : i] = False

    # --- Largest prime factor for every integer up to NMAX ---
    lpf = np.zeros(NMAX + 1, dtype=np.int32)
    
    # Stamp largest prime factor
    # Iterate through primes in increasing order.
    # The last prime that divides a number m will be its largest prime factor.
    primes = np.where(is_prime)[0]
    
    for p in primes:
        # Vectorized assignment: set lpf for all multiples of p to p
        # Since p increases, the last assignment wins, which is the largest prime factor.
        lpf[p : NMAX+1 : p] = p

    # --- Scan the whole window for the global minimum of max(lpf[n], lpf[n+1]) ---
    # We need n in [10000, 1000000]
    # Objective for n is max(lpf[n], lpf[n+1])
    # Create array of objectives for n from 10000 to N
    # lpf indices: n ranges from 10000 to N, so lpf[n] is lpf[10000:N+1]
    # lpf[n+1] is lpf[10001:N+2] -> wait, NMAX is N+1, so lpf has size NMAX+1 = N+2.
    # lpf[N+1] is valid.
    
    obj = np.maximum(lpf[10000:N+1], lpf[10001:N+2])
    
    # Find index of minimum value
    min_idx = np.argmin(obj)
    best_n = 10000 + min_idx
    
    return best_n
