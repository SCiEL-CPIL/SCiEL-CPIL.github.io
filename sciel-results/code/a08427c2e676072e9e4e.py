"""
Strategy: Localized Radial Basis Function (RBF) Network.

The previous strategy used a global Chebyshev polynomial basis. While orthogonal,
global bases can struggle with high-frequency or non-smooth targets due to the
Ritz phenomenon (oscillations) and potential ill-conditioning when the target
requires high-order terms. 

This strategy switches to a local basis: Gaussian RBFs centered on a fixed 3x3 grid
within the domain [-1, 1]^2. 
- 9 centers provide local flexibility.
- 1 constant term (bias) ensures the function can represent a non-zero mean.
- Total parameters: 10 (9 for RBF weights + 1 for bias).

The Gaussian kernel $K_i(x, y) = \exp(-\frac{(x-c_{xi})^2 + (y-c_{yi})^2}{2\sigma^2})$ 
is bounded and smooth. By fixing the centers and width $\sigma$, we create a well-behaved
non-linear regression problem. BFGS can optimize the linear combination weights 
effectively because the basis functions are localized, reducing the correlation 
between parameters compared to a global polynomial basis of similar order.
"""

import numpy as np
import time

_start_time = time.time()
_TIMEOUT_MARGIN = 148.0  # seconds, keep a safety margin before 150s kill

def _time_left():
    return _start_time + _TIMEOUT_MARGIN - time.time()

# Precompute centers for the 3x3 grid
# Centers at -1, 0, 1
centers = np.array([
    [-1.0, -1.0],
    [-1.0,  0.0],
    [-1.0,  1.0],
    [ 0.0, -1.0],
    [ 0.0,  0.0],
    [ 0.0,  1.0],
    [ 1.0, -1.0],
    [ 1.0,  0.0],
    [ 1.0,  1.0]
])

# Width parameter for Gaussian kernels.
# Chosen to provide overlap between neighbors but maintain locality.
# sigma = 0.5 is a reasonable starting point for domain size 2.
SIGMA = 0.5
SIGMA_SQ_2 = 2.0 * SIGMA * SIGMA

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters using a Gaussian RBF basis.
    
    params:
        params[0:9]: Weights for the 9 Gaussian RBFs centered on the 3x3 grid.
        params[9]:   Bias (constant term).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Check timeout (though this function is fast, it's good practice per contract)
    if _time_left() < 0.1:
        # Return a simple fallback if we are about to time out (unlikely for this simple math)
        return np.zeros_like(x)

    # Calculate squared distances from each point (x, y) to each center
    # x, y are shape (N,). centers are shape (9, 2).
    # We want dists to be shape (N, 9)
    
    # Efficient vectorized distance calculation
    # x[:, None] is (N, 1), centers[:, 0] is (9,) -> (N, 9)
    dx = x[:, None] - centers[:, 0]
    dy = y[:, None] - centers[:, 1]
    
    # Squared Euclidean distance
    dist_sq = dx**2 + dy**2
    
    # Gaussian kernel
    # K = exp(-dist_sq / (2 * sigma^2))
    # Shape: (N, 9)
    K = np.exp(-dist_sq / SIGMA_SQ_2)
    
    # Linear combination of RBFs
    # weights are params[0:9], shape (9,)
    # We need to sum K * weights over the center dimension (axis=1)
    # Result shape: (N,)
    rbf_sum = K @ params[:9]
    
    # Add bias
    bias = params[9]
    
    return rbf_sum + bias

# EVOLVE-BLOCK-END
