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
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Strategy: Direct search with on-the-fly totient calculation.
    The minimal n is expected to be small (in the hundreds or low thousands),
    so the loop terminates quickly without needing a large sieve table.
    
    We include a deadline guard to ensure we finish well within the 90s limit.
    If the search takes too long (which it shouldn't for the minimal n),
    we fall back to a known valid candidate.
    """
    start_time = time.time()
    deadline = start_time + 60  # Safety margin (90s limit, leave 30s buffer)
    
    # We search for the smallest n >= 1
    # Since we minimize n, we go upwards from 1
    # The answer is expected to be small, so we can start with a reasonable upper bound
    # and expand if needed, but 1000000 is the max allowed.
    
    # Optimization: We know that phi(n) must be divisible by 120 = 8 * 3 * 5.
    # This imposes constraints on the prime factorization of n.
    # However, for small n, direct computation is fast enough.
    
    # Start with a smaller search range first for speed
    # If not found, expand the range
    search_limit = 1000000
    
    for n in range(1, search_limit + 1):
        # Check deadline
        if time.time() > deadline:
            # Fallback: return a known valid candidate
            # 143 is known to be valid: phi(143) = phi(11*13) = 10*12 = 120
            return 143
        
        if _phi(n) % 120 == 0:
            return n
    
    # Should not reach here if 143 is valid and < 1000000
    # But if something went wrong, return a known valid number
    return 143
# EVOLVE-BLOCK-END
