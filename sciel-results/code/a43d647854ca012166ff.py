"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Powers of 2 from 2^0 to 2^8
    # 1, 2, 4, 8, 16, 32, 64, 128, 256
    # All are distinct and in [1, 256]
    # Superincreasing property ensures all subset sums are unique,
    # which implies sums of different cardinalities are distinct.
    return [1, 2, 4, 8, 16, 32, 64, 128, 256]

# EVOLVE-BLOCK-END
