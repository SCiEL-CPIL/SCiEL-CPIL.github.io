"""Find the smallest n in [1, 1000000] with phi(n) divisible by 120."""

def _totient(n):
    """Compute Euler's totient function phi(n) using trial division."""
    if n <= 1:
        return 1
    result = n
    x = n
    # Factor out 2
    if x % 2 == 0:
        result -= result // 2
        while x % 2 == 0:
            x //= 2
    # Check odd factors
    p = 3
    while p * p <= x:
        if x % p == 0:
            result -= result // p
            while x % p == 0:
                x //= p
        p += 2
    if x > 1:
        result -= result // x
    return result

def construct():
    """Return the smallest n in [1, 1000000] such that phi(n) is divisible by 120."""
    # Direct search from n=3 upward with optimized totient.
    # The answer is expected to be small (under a few thousand),
    # so this completes in well under 1 second.
    for n in range(3, 1000001):
        if _totient(n) % 120 == 0:
            return n
    return 14400  # Fallback (should never be reached)
