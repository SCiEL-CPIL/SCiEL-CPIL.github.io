"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 1. Define fixed RBF centers and widths
    # We have 10 parameters. We use 10 RBFs.
    # Centers evenly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Widths (sigma) chosen to provide local sensitivity.
    # Distance between centers is 4/9 ≈ 0.44.
    # A sigma of ~0.5 gives good overlap for smooth functions.
    sigma = 0.5
    
    # 2. Compute RBF activations
    # x shape: (N, 1) or (N,)
    # We need to ensure x is 1D for broadcasting or handle shapes carefully.
    # The evaluator passes x as an array. Let's assume 1D or flatten.
    x_flat = np.asarray(x).flatten()
    
    # Compute distances: |x_i - c_j|
    # x_flat: (N, 1)
    # centers: (1, 10)
    dist_sq = (x_flat[:, np.newaxis] - centers[np.newaxis, :]) ** 2
    
    # RBF: exp(-dist_sq / (2 * sigma^2))
    # Note: Standard Gaussian is exp(-0.5 * ((x-mu)/sigma)^2) = exp(-dist_sq / (2*sigma^2))
    rbf_values = np.exp(-dist_sq / (2.0 * sigma * sigma))
    
    # 3. Linear combination
    # rbf_values shape: (N, 10)
    # params shape: (10,)
    # result shape: (N,)
    return rbf_values @ params
# EVOLVE-BLOCK-END
