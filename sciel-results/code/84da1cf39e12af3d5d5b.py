"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns a set of distinct positive integers <= 512 such that no element
    is the arithmetic mean of any subset of at least two other elements.
    
    We use the set of powers of 2: {1, 2, 4, ..., 512}.
    Proof:
    Let S = {2^0, 2^1, ..., 2^9}.
    Suppose x = 2^j in S is the mean of a subset T of k >= 2 other elements.
    Then sum(T) = k * 2^j.
    T consists of distinct powers of 2, none equal to 2^j.
    The sum of k distinct powers of 2 has a binary representation with exactly k bits set to 1.
    The number k * 2^j has a binary representation with popcount(k) bits set to 1.
    For the equality to hold, we must have popcount(k) = k.
    This equation holds only for k = 1.
    Since k >= 2, no such subset T exists.
    Thus, the set of powers of 2 is valid.
    Size is 10 (2^0 to 2^9).
    """
    selected = []
    x = 1
    while x <= 512:
        selected.append(x)
        x *= 2
    return selected

# EVOLVE-BLOCK-END
