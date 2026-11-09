"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates y using a Gaussian Radial Basis Function (RBF) expansion.
    
    Structural Change: 
    Instead of a mixed polynomial/exponential/trig basis (current) or pure polynomials (inspiration 3),
    this uses a set of 10 Gaussian kernels centered evenly across the domain.
    This provides a flexible, local approximation capability that can capture
    non-polynomial features (like peaks or valleys) better than global polynomials,
    while remaining a linear model in the parameters for stable BFGS convergence.
    
    y(x) = sum_{i=0}^{9} params[i] * exp(-((x - centers[i])^2) / (2 * sigma^2))
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 10.0 # 10s is plenty for this computation
    
    # Domain is [0.1, 6]
    # Define 10 centers evenly spaced in the domain
    centers = np.linspace(0.1, 6.0, 10)
    
    # Choose a sigma that provides good overlap but distinct peaks.
    # Distance between centers is approx (6-0.1)/9 = 0.65.
    # Sigma = 0.5 * spacing is a common heuristic, so ~0.32.
    # However, to be safe and smooth, let's use sigma = 0.6.
    sigma = 0.6
    sigma_sq = 2.0 * sigma * sigma
    
    # Compute the RBF basis values
    # x shape: (N,)
    # centers shape: (10,)
    # We want a matrix of shape (N, 10) where basis[j, i] = phi(x[j], centers[i])
    
    # x[:, None] is (N, 1)
    # centers[None, :] is (1, 10)
    # diff is (N, 10)
    diff = x[:, None] - centers[None, :]
    
    # Compute Gaussian
    # exp(-diff^2 / (2*sigma^2))
    basis = np.exp(-(diff * diff) / sigma_sq)
    
    # Linear combination
    # basis shape (N, 10), p shape (10,) -> result (N,)
    y = basis @ p
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(y)):
        # If numerical issues arise, return zeros or best effort
        # But with bounded exp, this is unlikely.
        y = np.nan_to_num(y)
        
    return y
# EVOLVE-BLOCK-END
