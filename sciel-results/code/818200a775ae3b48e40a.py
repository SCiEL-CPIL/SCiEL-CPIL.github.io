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
3. Linear in Parameters: The model is linear in the weights, so BFGS will find the
   global optimum for this specific basis without getting stuck in local minima.
4. Uniform Coverage: The centers are uniformly distributed, ensuring consistent
   approximation quality across the entire domain [-3, 3], unlike the current program
   which has sparse coverage in the tails.

The width (sigma) of the Gaussians is chosen to be roughly the distance between
adjacent centers (6/9 = 0.66), ensuring good overlap and smoothness.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-compute centers and widths for the RBF network
# 10 centers evenly spaced in [-3, 3]
# np.linspace(-3, 3, 10)
centers = np.linspace(-3.0, 3.0, 10)
# Width: roughly the step size, maybe a bit wider for smoothness
# Step = 6 / 9 = 0.666...
# Let's use sigma = 0.7
sigma = 0.7
inv_2_sigma_sq = 1.0 / (2.0 * sigma**2)

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the RBF network output.
    
    Args:
        x: Input array of shape (N,)
        params: Weights array of shape (10,)
    
    Returns:
        Prediction array of shape (N,)
    """
    x = np.asarray(x).ravel()
    
    # Compute distances from each x to each center
    # x is (N,), centers is (10,)
    # We want a matrix of shape (N, 10) where M[i, j] = (x[i] - centers[j])**2
    # Using broadcasting: x[:, np.newaxis] - centers[np.newaxis, :]
    diffs = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Compute Gaussian values: exp(-0.5 * (diff/sigma)^2)
    # This is efficient and vectorized
    rbf_values = np.exp(inv_2_sigma_sq * -(diffs**2))
    
    # Linear combination: y = rbf_values @ params
    # rbf_values is (N, 10), params is (10,)
    # Result is (N,)
    return rbf_values @ params

# EVOLVE-BLOCK-END
