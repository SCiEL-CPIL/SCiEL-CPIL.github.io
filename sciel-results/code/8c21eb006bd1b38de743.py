def construct():
    """Return a square-free integer (product of first k primes) <= 10^18
    that maximizes tau_perp(n), the number of adjacent coprime divisor pairs."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n = 1
    for p in primes:
        n *= p
    # n should be <= 10^18; verify
    assert 2 <= n <= 10**18, f"n={n} out of range"
    return n
