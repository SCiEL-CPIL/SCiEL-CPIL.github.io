"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Use pairs that sum to the same value to force many pair-sum collisions
    # {0,22}, {1,21}, {2,20}, {3,19}, {4,18}, {5,17} all sum to 22
    # This creates massive additive collisions, making it hard to find large Sidon subsets
    return [0, 22, 1, 21, 2, 20, 3, 19, 4, 18, 5, 17]
# EVOLVE-BLOCK-END
