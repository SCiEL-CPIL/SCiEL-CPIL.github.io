"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n*(n+1).
    Uses sieve-based precomputation of largest prime factors (LPF) for O(1) lookup.
    """
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    N_MAX = 1000000
    N_MIN = 10000
    
    # Precompute largest prime factor for all numbers up to N_MAX + 1
    # Using a sieve approach: for each prime p, mark multiples with p as their LPF
    lpf = list(range(N_MAX + 2))  # lpf[i] will store the largest prime factor of i
    
    # Sieve of Eratosthenes variant to compute LPF
    # For each prime p, update lpf[p*k] = max(lpf[p*k], p)
    # More efficiently: iterate primes and set lpf for multiples
    is_prime = [True] * (N_MAX + 2)
    is_prime[0] = is_prime[1] = False
    
    # Standard sieve to find primes
    for i in range(2, int(math.isqrt(N_MAX + 1)) + 1):
        if is_prime[i]:
            for j in range(i * i, N_MAX + 2, i):
                is_prime[j] = False
    
    # Now compute LPF: for each prime, mark its multiples
    # lpf[i] starts as i (which is correct for primes)
    # For composite i, lpf[i] should be the largest prime factor
    # We can do: for each prime p, for each multiple m of p, lpf[m] = max(lpf[m], p)
    # But a more efficient way: lpf[m] = p if p is the largest prime factor
    # Actually, let's use: for each number, its LPF can be found by dividing out smallest prime factors
    # Alternative: lpf[i] = i if prime, else lpf[i // spf[i]] where spf is smallest prime factor
    # Let's compute smallest prime factor first, then derive LPF
    
    spf = list(range(N_MAX + 2))  # smallest prime factor
    for i in range(2, int(math.isqrt(N_MAX + 1)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N_MAX + 2, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Now compute LPF using SPF
    # lpf[i] = max(spf[i], lpf[i // spf[i]])
    lpf = [0] * (N_MAX + 2)
    lpf[0] = 0
    lpf[1] = 1
    for i in range(2, N_MAX + 2):
        p = spf[i]
        rest = i // p
        if rest == 1:
            lpf[i] = p
        else:
            lpf[i] = max(p, lpf[rest])
    
    # Check time
    if time.time() > deadline:
        # Fallback: return a reasonable value
        return 10080  # 10080 = 2^5 * 3^2 * 5 * 7, smooth
    
    # Now search for n in [N_MIN, N_MAX] that minimizes max(lpf[n], lpf[n+1])
    best_n = N_MIN
    best_val = max(lpf[N_MIN], lpf[N_MIN + 1])
    
    for n in range(N_MIN, N_MAX + 1):
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
        
        # Check time periodically
        if n % 100000 == 0 and time.time() > deadline:
            break
    
    return best_n

# EVOLVE-BLOCK-END
