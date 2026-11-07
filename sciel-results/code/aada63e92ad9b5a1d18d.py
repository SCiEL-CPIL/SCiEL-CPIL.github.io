"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

_start = time.time()
_DEADLINE = 85  # safety margin before 90s kill

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _largest_prime_power_le(p, max_val):
    """For prime p, return the largest power p^k <= max_val."""
    result = 1
    while result * p <= max_val:
        result *= p
    return result

def construct():
    max_val = 256
    # Generate all primes up to max_val
    primes = [p for p in range(2, max_val + 1) if _is_prime(p)]
    
    # For each prime, compute the largest power <= max_val
    result = [1]
    for p in primes:
        pp = _largest_prime_power_le(p, max_val)
        result.append(pp)
    
    # Convert to Python int and ensure it's a list
    result = [int(x) for x in result]
    
    # Verify pairwise coprime (sanity check, cheap for this size)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # Fallback: return just primes (which are pairwise coprime)
                return [1] + [p for p in primes]
    
    return result

# EVOLVE-BLOCK-END
