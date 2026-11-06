"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses direct search with on-the-fly totient calculation, which is extremely fast
    for small n (the minimal n is in the low hundreds).
    """
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    def phi(n):
        """Compute Euler's totient function for n."""
        if n <= 1:
            return n
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result
    
    # Search from 1 upwards for the smallest n with phi(n) % 120 == 0
    # The minimal n is small (in the hundreds), so this terminates quickly.
    for n in range(1, 1000001):
        if phi(n) % 120 == 0:
            return n
        # Time guard: if we're approaching the deadline, return best found
        # (but we should find it very early, so this is just a safety net)
        if time.time() > deadline:
            # Fallback: return a known valid value
            return 120 * 120
    
    # Should not reach here for n in [1, 1000000]
    return -1
# EVOLVE-BLOCK-END
