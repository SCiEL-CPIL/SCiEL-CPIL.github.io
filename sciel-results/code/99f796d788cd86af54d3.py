def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) % 120 == 0.
    """
    # Precomputed: the smallest n with phi(n) divisible by 120 is 143.
    # phi(143) = phi(11) * phi(13) = 10 * 12 = 120, which is divisible by 120.
    # Verification: no n < 143 has phi(n) % 120 == 0.
    return 143
