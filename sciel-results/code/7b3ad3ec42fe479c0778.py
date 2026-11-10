# EVOLVE-BLOCK-START
def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) % 120 == 0.
    Uses trial division for phi(n) with a small search bound since the answer is known to be small.
    """
    def phi(n):
        """Compute Euler's totient function using trial division."""
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

    # Search for the smallest n with phi(n) divisible by 120
    # The answer is known to be small; search up to a safe bound
    for n in range(1, 200):
        if phi(n) % 120 == 0:
            return n

    # Fallback: if not found in small range (should not happen), use a larger search
    for n in range(200, 1000001):
        if phi(n) % 120 == 0:
            return n

    return 1
# EVOLVE-BLOCK-END
