"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
def construct():
    # A single spike maximizes the ratio of L2 norm squared to L1 norm times autoconvolution sup norm
    # for non-negative signals.
    # J(spike) = 1.0
    # J(uniform) = 1/N = 0.02
    h = [0.0] * N
    h[0] = 1.0
    return h
# EVOLVE-BLOCK-END
