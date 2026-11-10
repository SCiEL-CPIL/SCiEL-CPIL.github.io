"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Uses a direct search with trial-division totient computation, which is sufficient
    since the answer is small and the computation is trivial.
    """
    def totient(n):
        """Compute Euler's totient of n by trial division."""
        result = n
        m = n
        d = 2
        while d * d <= m:
            if m % d == 0:
                while m % d == 0:
                    m //= d
                result -= result // d
            d += 1
        if m > 1:
            result -= result // m
        return result

    for n in range(1, 1000001):
        if totient(n) % 120 == 0:
            return n
    
    # Fallback (should not happen)
    return 1000000
# EVOLVE-BLOCK-END
