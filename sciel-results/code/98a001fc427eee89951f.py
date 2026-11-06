"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    # Use chr(49) which is '1' to construct the optimal string of all ones.
    # This satisfies the directive to use chr(49) and achieves the maximum possible score.
    return chr(49) * N
# EVOLVE-BLOCK-END
