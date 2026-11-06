"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def coefficients(stages):
    """Symmetric 12-stage product formula with equal weights for A and B at each stage."""
    a = [1.0 / stages] * stages
    b = [1.0 / stages] * stages
    return a, b
# EVOLVE-BLOCK-END
