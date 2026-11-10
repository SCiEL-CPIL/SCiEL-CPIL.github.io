"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums must differ.
    
    Strategy:
    Use the set of the largest 31 integers: {226, 227, ..., 256}.
    We verified that for an arithmetic progression {a, ..., a+n-1}, the condition
    min_sum(k) > max_sum(k-1) holds for all k if n^2 < 4*(max_val - a + 1).
    For n=31 and a=226 (max=256), 31^2 = 961 < 4*31 = 124? No.
    The condition derived was n^2 < 4*Range where Range is the span?
    Let's re-verify the condition for {226..256}.
    n=31.
    Condition: k^2 - nk + 256 > 0 was for {256-n+1..256}.
    Here a=226.
    The condition was k^2 - nk + 256 > 0 assuming the set ends at 256.
    Wait, the constant 256 in the inequality $k^2 - nk + 256 > 0$ came from the max value.
    If the set is {226..256}, the max value is 256.
    So the condition holds.
    Min at k=15.5.
    f(15) = 225 - 465 + 256 = 16 > 0.
    f(16) = 256 - 496 + 256 = 16 > 0.
    So the sufficient condition holds.
    Thus, the set is valid.
    """
    # The set of the 31 largest integers
    return list(range(226, 257))

# EVOLVE-BLOCK-END
