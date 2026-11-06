"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The set of integers from 171 to 256 inclusive.
    # For any a, b, c in this set with a < b <= c:
    # b+c >= (a+1) + (a+1) = 2a+2.
    # The next multiple of a is 3a.
    # If 3a > 2*max(S) = 512, then no multiple of a can be formed by b+c.
    # 3*171 = 513 > 512.
    # Thus, a does not divide b+c for any such triple.
    # Cardinality is 256 - 171 + 1 = 86.
    return list(range(171, 257))
# EVOLVE-BLOCK-END
