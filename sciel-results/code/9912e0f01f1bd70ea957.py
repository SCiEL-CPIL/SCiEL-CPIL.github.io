"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_cost(f):
    """
    Compute the cost J = max_k |AC(k)| / (sum(f))^2.
    Returns infinity if sum(f) is 0.
    """
    s = np.sum(f)
    if s == 0:
        return float('inf')
    
    # Compute autocorrelation
    ac = np.correlate(f, f, mode='full')
    
    max_abs_ac = np.max(np.abs(ac))
    
    return max_abs_ac / (s * s)

def construct():
    """
    Construct a signed step function that minimizes the maximum absolute
    autoconvolution divided by the squared integral.
    
    Strategy:
    The cost is max(|AC(k)|) / (sum(f))^2.
    For binary sequences f in {-1, 1}, AC(0) = sum(f^2) = N.
    Thus max(|AC(k)|) >= N.
    The sum(f) is maximized when all f_i = 1, giving sum(f) = N.
    In this case, AC(k) = N - |k|, so max(|AC(k)|) = N (at k=0).
    Cost = N / N^2 = 1/N.
    Any other binary sequence has sum(f) < N, so (sum(f))^2 < N^2.
    Since max(|AC(k)|) >= N, the cost is >= N / (sum(f))^2 > N / N^2 = 1/N.
    Therefore, the all-ones sequence is the unique global optimum for binary inputs.
    """
    return np.ones(N)
# EVOLVE-BLOCK-END
