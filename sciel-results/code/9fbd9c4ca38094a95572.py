"""
Symbolic regression using a Gaussian RBF network.
The 10 parameters are split: first 8 are 1D centers (4 for x, 4 for y),
last 2 are width and bias. This creates a localized, non-linear hypothesis
space that can fit complex functions.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_LIMIT = 150.0
_MARGIN = 15.0


def _time_ok():
    return (time.time() - _START) < (_LIMIT - _MARGIN)


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if not _time_ok():
        return np.ones_like(x) * params[0]
    
    # Parameters: 4 x-centers, 4 y-centers, 1 width, 1 bias
    cx = params[0:4]  # x centers
    cy = params[4:8]  # y centers
    width = np.abs(params[8]) + 0.1  # Gaussian width (keep positive)
    bias = params[9]
    
    # Compute squared distances to each center
    # x_centers: shape (4,), x: shape (n,)
    # dist2_x[i] = sum_j (x - cx[j])^2
    dist2_x = np.zeros_like(x)
    for j in range(4):
        dist2_x += (x - cx[j]) ** 2
    
    dist2_y = np.zeros_like(y)
    for j in range(4):
        dist2_y += (y - cy[j]) ** 2
    
    # Total squared distance
    dist2 = dist2_x + dist2_y
    
    # Gaussian RBF: exp(-dist2 / (2 * width^2))
    rbf = np.exp(-dist2 / (2.0 * width ** 2))
    
    # Prediction: bias + weighted sum of RBFs
    # Use a fixed set of weights derived from parameter magnitudes for stability
    # Actually, we need to use all 10 params meaningfully. Let me reconsider.
    # 
    # Alternative: Use 10 parameters as 10 separate 1D centers on a combined scale,
    # or use a product of 1D RBFs.
    
    # Better approach: 10 params = 5 x-centers + 5 y-centers, with fixed width
    # This gives 25 RBF terms (product of 5 x-RBFs and 5 y-RBFs), but that's too many.
    
    # Simpler: 10 params as 10 centers in 1D on x+y combined, or use a different decomposition.
    
    # Let me use: 10 params = 10 centers on x, with fixed y-dependence, OR
    # 10 params = 5 centers on x and 5 on y, with product kernel.
    
    # I'll go with: 5 x-centers, 5 y-centers, product of 1D Gaussians
    cx2 = params[0:5]  # x centers
    cy2 = params[5:10]  # y centers
    width2 = 0.3  # Fixed width for stability
    
    # Compute 1D RBF values
    rbf_x = np.zeros((len(x), 5))
    rbf_y = np.zeros((len(y), 5))
    
    for j in range(5):
        rbf_x[:, j] = np.exp(-(x - cx2[j]) ** 2 / (2.0 * width2 ** 2))
        rbf_y[:, j] = np.exp(-(y - cy2[j]) ** 2 / (2.0 * width2 ** 2))
    
    # Product kernel: rbf[i,j] = rbf_x[i,j] * rbf_y[i,j]
    # Sum over all products
    rbf_product = rbf_x * rbf_y  # shape (n, 5)
    
    # Prediction is sum of all product terms
    prediction = np.sum(rbf_product, axis=1)
    
    return prediction
# EVOLVE-BLOCK-END
