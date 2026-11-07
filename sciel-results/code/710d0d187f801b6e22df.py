import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # 1. Base LCB Score
    # For minimization, we want low mu.
    # Score = -mu + beta * sigma
    beta = 2.5
    
    scores = -mu + beta * sigma
    
    # 2. Feature Space Diversity Penalty
    # Normalize mu and sigma to [0, 1] for the batch
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    
    # Avoid division by zero
    mu_range = mu_max - mu_min if mu_max > mu_min else 1.0
    sigma_range = sigma_max - sigma_min if sigma_max > sigma_min else 1.0
    
    mu_norm = (mu - mu_min) / mu_range
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Combine into 2D points
    points = np.column_stack((mu_norm, sigma_norm))
    
    # Calculate pairwise squared distances
    # This is O(N^2), N=2000 -> 4M ops. Fast enough.
    dists = np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :])**2, axis=2)
    
    # Gaussian Kernel Density Estimate
    # Bandwidth h. Since normalized to [0,1], h=0.1 seems reasonable.
    h = 0.1
    kernel = np.exp(-dists / (2 * h**2))
    
    # Density for each point i is sum of kernels to all j != i
    # Subtract diagonal (self-distance is 0, kernel is 1)
    np.fill_diagonal(kernel, 0)
    density = np.sum(kernel, axis=1)
    
    # Normalize density to [0, 1] range for consistent penalty
    d_max = np.max(density)
    if d_max > 0:
        density = density / d_max
        
    # Penalty term
    lambda_penalty = 0.5
    scores = scores - lambda_penalty * density
    
    return scores
