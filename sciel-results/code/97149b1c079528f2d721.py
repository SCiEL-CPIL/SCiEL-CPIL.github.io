"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def _phi(n):
    """Compute Euler's totient function for n."""
    if n == 1:
        return 1
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

def construct():
    # Search for the smallest n in [1, 1000000] such that phi(n) is divisible by 120
    target = 120
    for n in range(1, 1000001):
        if _phi(n) % target == 0:
            return n
    return 1  # Fallback, though 143 is guaranteed to be found

# EVOLVE-BLOCK-END
