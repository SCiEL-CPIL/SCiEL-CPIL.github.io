"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # RBF basis with fixed centers
    # This approach uses local kernels which may be more robust for
    # functions with sharp features or non-linearities that global
    # polynomials struggle with.
    
    # Define 10 centers uniformly spaced in [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Compute squared distances between x and each center
    # x shape: (N,), centers shape: (10,)
    # dist_sq shape: (N, 10)
    dist_sq = (x[:, None] - centers[None, :]) ** 2
    
    # Gaussian RBF with beta = 2.0 (sigma = 0.5)
    # This provides a balance between local flexibility and global smoothness
    K = np.exp(-2.0 * dist_sq)
    
    # Linear combination of RBFs
    result = K @ params
    
    return result
# EVOLVE-BLOCK-END
