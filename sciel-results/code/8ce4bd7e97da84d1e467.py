"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a signed step function that minimizes the maximum absolute
    autoconvolution divided by the squared integral.
    
    Strategy:
    The objective is J = max_k |AC(k)| / (sum(f))^2.
    For a sequence of +/- 1, AC(0) = N.
    Thus J >= N / (sum(f))^2.
    To minimize J, we must maximize sum(f).
    The maximum sum for a non-constant sequence is N-2 (one -1, rest 1s).
    This yields J approx 1/N, which is significantly better than the
    local search result which typically has sum near 0.
    """
    f = np.ones(N)
    f[0] = -1.0
    return f
# EVOLVE-BLOCK-END
