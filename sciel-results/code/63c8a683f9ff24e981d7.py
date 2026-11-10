"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Choose 12 distinct positive roots clustered near the origin.
    # Smaller roots generally lead to a smaller 'B' (largest valid root of q),
    # which minimizes the score B / (2*pi).
    # We use a linear spacing starting at 0.1 to avoid 0 and ensure distinctness.
    z = [0.1 + 0.1 * i for i in range(12)]
    return z
# EVOLVE-BLOCK-END
