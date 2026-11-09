"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The set {63, ..., 128} has size 66.
    # Min sum of two distinct elements is 63+64=127.
    # Sums in the set are only 127 and 128.
    # A forbidden triple requires 3 distinct sums in the set.
    # Since only 2 sums are possible, no forbidden triple exists.
    # This beats the baseline of 64 (range(65, 129)).
    return list(range(63, 129))
# EVOLVE-BLOCK-END
