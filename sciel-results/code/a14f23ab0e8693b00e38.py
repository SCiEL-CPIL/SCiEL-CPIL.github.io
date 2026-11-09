"""
Strategy: Dense Uniform Radial Basis Function (RBF) Network.

Unlike the current program's mixed basis (polynomials, trig, sparse RBFs), which relies
on the evaluator's BFGS to navigate a potentially ill-conditioned or redundant feature
space, this program uses a dense, uniform grid of 10 Gaussian RBFs centered at evenly
spaced points in [-3, 3].

Key advantages:
1. Universal Approximation: A dense RBF network is a universal approximator. It can
   fit smooth functions of arbitrary complexity (within the resolution of the centers)
   much better than a fixed set of global polynomials or trig functions.
2. Localized Features: RBFs are localized, meaning the parameters are less correlated
   than high-degree polynomials, leading to better-conditioned optimization for BFGS.
3. Linear in Parameters: The model is linear in the params, so BFGS will converge
   reliably to the least-squares solution for this specific basis, avoiding local minima
   that might trap a non-linear model.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute the centers and the kernel matrix structure?
# No, x is passed in each call. But we can pre-compute the centers.
# Domain is [-3, 3]. We have 10 parameters.
# Let's place centers at linspace(-3, 3, 10).
# Width (sigma) needs to be chosen carefully.
# If sigma is too small, the matrix is ill-conditioned (basis functions are nearly orthogonal but sparse coverage).
# If sigma is too large, it behaves like a low-order polynomial (global).
# A good heuristic for [-3, 3] with 10 points:
# Spacing is 6/9 = 0.66.
# Sigma around 0.5 to 1.0 usually works well. Let's try 0.8.

CENTERS = np.linspace(-3.0, 3.0, 10)
SIGMA = 0.8

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a linear combination of Gaussian RBFs.
    
    Args:
        x: Input array of shape (N,)
        params: Array of 10 coefficients.
        
    Returns:
        Prediction array of shape (N,)
    """
    # Ensure x is a 1D numpy array for consistent operations
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # Compute distances from each x to each center
    # x shape: (N, 1)
    # CENTERS shape: (1, 10)
    # dist_sq shape: (N, 10)
    dist_sq = (x[:, np.newaxis] - CENTERS[np.newaxis, :]) ** 2
    
    # Compute Gaussian kernels: exp(-dist_sq / (2 * sigma^2))
    # sigma^2 = 0.64
    # 2 * sigma^2 = 1.28
    kernels = np.exp(-dist_sq / 1.28)
    
    # Linear combination: y = kernels @ params
    # kernels shape: (N, 10)
    # params shape: (10,)
    # result shape: (N,)
    y = kernels @ params
    
    return y

# EVOLVE-BLOCK-END
