"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 600

# EVOLVE-BLOCK-START
def construct():
    # Based on the analysis that minimizing the ratio of max autoconvolution 
    # to the square of the max function value (L-infinity normalization) 
    # favors sparsity. A single non-zero bin minimizes the sum of squares 
    # relative to the peak height.
    heights = [0.0] * N
    # Place a spike in the center
    heights[N // 2] = 1.0
    return heights
# EVOLVE-BLOCK-END
