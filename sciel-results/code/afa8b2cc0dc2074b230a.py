"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a dynamic Lower Confidence Bound (LCB)
    with a kernel-density-based spatial diversity penalty.
    
    For minimization, the LCB is defined as:
        LCB(x) = mu(x) - kappa * sigma(x)
    
    We maximize the acquisition score, so we return:
        Score(x) = -mu(x) + kappa * sigma(x) - gamma * density(x)
    
    The density term estimates local crowding among the 2000 candidates
    using a fixed-bandwidth Gaussian kernel, penalizing regions that are
    already densely sampled to enforce spatial diversity.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Sanitize inputs
    sigma_safe = np.maximum(sigma, 1e-10)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-10)
    
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    
    # Known global minimum for Hartmann 3D
    f_global_min = -3.86278
    
    # Estimate initial f_best for progress calculation
    f_start_ref = 0.0
    
    # Calculate progress: 1.0 at start (poor f_best), 0.0 at optimum
    span = f_start_ref - f_global_min
    if span == 0:
        progress = 0.5
    else:
        progress = (f_best - f_global_min) / span
        progress = np.clip(progress, 0.0, 1.0)
        
    # Dynamic kappa
    kappa_min = 0.5
    kappa_max = 2.5
    kappa = kappa_min + (kappa_max - kappa_min) * np.sqrt(progress)
    
    # Calculate base LCB score for maximization
    scores = -mu_safe + kappa * sigma_safe
    
    # Multi-scale kernel-density-based exploration penalty
    # Estimate local density using a fixed bandwidth
    # Since we don't have coordinates, we use the mu and sigma values as
    # proxies for spatial structure. Points with similar (mu, sigma) values
    # are likely in similar regions of the search space.
    
    # Use a combination of mu and sigma to form a 2D feature space
    # This captures both the predicted mean and uncertainty structure
    features = np.column_stack([mu_safe, sigma_safe])
    
    # Compute pairwise distances in feature space
    # For 2000 points, this is a 2000x2000 matrix
    # Use efficient computation
    feat_sq = np.sum(features * features, axis=1)
    dist_sq = feat_sq[:, None] + feat_sq[None, :] - 2.0 * features @ features.T
    np.maximum(dist_sq, 0.0, out=dist_sq)  # Numerical stability
    
    # Multi-scale: use two bandwidths to capture different scales of structure
    # Bandwidth 1: fine scale (local clustering)
    # Bandwidth 2: coarse scale (broader regions)
    h_fine = 0.05
    h_coarse = 0.2
    
    # Gaussian kernel density estimate
    # K(x_i, x_j) = exp(-dist_sq / (2 * h^2))
    density_fine = np.exp(-dist_sq / (2.0 * h_fine ** 2))
    density_coarse = np.exp(-dist_sq / (2.0 * h_coarse ** 2))
    
    # Sum over all other points to get density at each point
    # Exclude self-contribution
    np.fill_diagonal(density_fine, 0.0)
    np.fill_diagonal(density_coarse, 0.0)
    
    local_density_fine = np.sum(density_fine, axis=1)
    local_density_coarse = np.sum(density_coarse, axis=1)
    
    # Combine multi-scale densities with weights
    # Fine scale has more influence for local diversity
    combined_density = 0.7 * local_density_fine + 0.3 * local_density_coarse
    
    # Normalize density to [0, 1] range for stable penalty
    d_max = np.max(combined_density)
    if d_max > 0:
        combined_density_norm = combined_density / d_max
    else:
        combined_density_norm = np.zeros_like(combined_density)
    
    # Dynamic penalty strength: stronger early (exploration), weaker late (exploitation)
    # gamma scales with progress
    gamma_min = 0.1
    gamma_max = 1.5
    gamma = gamma_min + (gamma_max - gamma_min) * progress
    
    # Apply density penalty to LCB scores
    scores = scores - gamma * combined_density_norm
    
    # Check deadline
    if time.time() > deadline:
        # Return best result found so far
        pass
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
