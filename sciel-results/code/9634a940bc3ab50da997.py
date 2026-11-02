"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct a signed equal-width step function using a heuristic local search
    (Hill Climbing) approach.
    
    The objective is to minimize M = max_k |C_k| / S^2.
    We initialize with a constant sequence (known to be a strong candidate)
    and attempt to improve it by flipping individual elements.
    
    Note: For real-valued sequences, the constant sequence is the global
    minimum of the bound derived from Cauchy-Schwarz. This search verifies
    this locally and ensures stability.
    """
    # Initialize with a constant sequence
    x = np.ones(N)
    
    # Calculate initial metric
    def calc_metric(seq):
        S = np.sum(seq)
        if S == 0:
            return float('inf')
        # Autoconvolution
        # Using FFT for efficiency, though N=400 is small enough for direct
        # or np.convolve. np.convolve is 'full' mode, we need 'same' or specific lags.
        # C_k for k in 0..N-1 is sum_{i=0}^{N-1-k} x_i x_{i+k}
        # We need max absolute value.
        c = np.correlate(seq, seq, mode='full')
        # 'full' mode length is 2N-1. Center is N-1.
        # We consider all lags.
        max_c = np.max(np.abs(c))
        return max_c / (S * S)

    current_score = calc_metric(x)
    
    # Simple Hill Climbing
    rng = np.random.default_rng(42)
    for _ in range(50):
        # Pick a random index to flip
        idx = rng.integers(0, N)
        old_val = x[idx]
        x[idx] = -old_val
        
        new_score = calc_metric(x)
        
        if new_score < current_score:
            current_score = new_score
        else:
            # Revert
            x[idx] = old_val
            
    return x
# EVOLVE-BLOCK-END
