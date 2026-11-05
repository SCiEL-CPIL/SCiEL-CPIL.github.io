"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The set {171, ..., 256} is valid.
    # For any a in this set, 3a >= 513 > 512 (max sum b+c).
    # Thus a cannot divide b+c for b,c > a.
    # Size is 256 - 171 + 1 = 86.
    return list(range(171, 257))
# EVOLVE-BLOCK-END
