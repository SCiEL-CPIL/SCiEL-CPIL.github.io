"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 600

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a step function with 600 intervals on [-1/4, 1/4].
    To minimize the normalized maximum of the autoconvolution (ratio of max autoconvolution to L2 norm squared),
    we use a pattern of disjoint spikes.
    Since the autoconvolution of a single bin has a support width of 2 bins,
    placing spikes at every other bin (distance 2) ensures their autoconvolutions are disjoint.
    This maximizes the number of spikes (300) while keeping the peak value constant relative to the total energy.
    The resulting score is approximately 1/300.
    """
    h = [0.0] * N
    # Place spikes at even indices: 0, 2, 4, ..., 598
    for i in range(0, N, 2):
        h[i] = 1.0
    return h
# EVOLVE-BLOCK-END
