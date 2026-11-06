"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _phi(n):
    """Compute Euler's totient function for n using trial division."""
    if n == 1:
        return 1
    result = n
    p = 2
    nn = n
    while p * p <= nn:
        if nn % p == 0:
            while nn % p == 0:
                nn //= p
            result -= result // p
        p += 1
    if nn > 1:
        result -= result // nn
    return result

def construct():
    # Find the smallest n >= 1 such that phi(n) % 120 == 0
    # The minimal n is small (in the hundreds), so direct search is fast.
    # Use a deadline guard for safety.
    deadline = time.time() + 85.0
    
    # First, do a quick search up to a reasonable bound
    # We know from number theory that the answer is small
    for n in range(1, 10000):
        if time.time() > deadline:
            return 143
        if _phi(n) % 120 == 0:
            return n
    
    # If not found in the quick range (should not happen), do a full search
    # with periodic deadline checks
    for n in range(10000, 1000001):
        if time.time() > deadline:
            return 143
        if _phi(n) % 120 == 0:
            return n
    
    # Fallback: return a known valid candidate
    return 143
# EVOLVE-BLOCK-END
