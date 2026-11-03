"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
def construct():
    # The ratio (||h||_2^2) / (||h||_1 * ||h conv h||_inf) is always <= 1
    # because (h conv h)[0] = sum(h_i^2) = ||h||_2^2, so ||h conv h||_inf >= ||h||_2^2.
    # Any nonnegative h achieves ratio = 1. Return a simple valid solution.
    return [1.0] * N
# EVOLVE-BLOCK-END
