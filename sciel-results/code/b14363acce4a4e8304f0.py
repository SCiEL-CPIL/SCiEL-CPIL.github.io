"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.spatial import cKDTree
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Styblinski Tang 2D.
    Uses KD-tree based local baseline for Expected Improvement to escape local minima.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Safety: check time budget
    deadline = 325.0  # 5 seconds safety margin from 330s limit
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: simple EI
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # KD-tree nearest-neighbor averaging in (mu, sigma) space for local baseline
    # This provides a more principled local baseline than global f_best,
    # better targeting basin boundaries where sigma is high relative to local predictions
    points = np.column_stack([mu, sigma_safe])
    
    try:
        tree = cKDTree(points)
        # Query for k nearest neighbors (excluding self)
        k_neighbors = min(10, n - 1)
        if k_neighbors > 0:
            dists, indices = tree.query(points, k=k_neighbors + 1)
            # dists has shape (n, k+1), indices has shape (n, k+1)
            # Take the first k neighbors (excluding self)
            neighbor_mu = points[indices[:, 1:k_neighbors+1], 0]  # (n, k)
            neighbor_sigma = points[indices[:, 1:k_neighbors+1], 1]  # (n, k)
            neighbor_dists = dists[:, 1:k_neighbors+1]  # (n, k)
            
            # Weighted average using inverse distance weighting
            # Add small epsilon to avoid division by zero
            inv_dists = 1.0 / (neighbor_dists + 1e-8)
            weights = inv_dists / (inv_dists.sum(axis=1, keepdims=True) + 1e-12)
            
            local_mu_baseline = (weights * neighbor_mu).sum(axis=1)
            local_sigma_baseline = (weights * neighbor_sigma).sum(axis=1)
            
            # Use local baseline for EI computation
            # The baseline is the weighted average of neighbor predictions
            # We want to improve upon this local baseline
            delta = local_mu_baseline - mu  # positive if current point is better than local baseline
            z_local = delta / sigma_safe
            
            # Expected Improvement relative to local baseline
            ei_local = delta * norm.cdf(z_local) + sigma_safe * norm.pdf(z_local)
            ei_local = np.maximum(ei_local, 0.0)
        else:
            # Fallback to global EI
            z = (f_best - mu) / sigma_safe
            ei_local = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
            ei_local = np.maximum(ei_local, 0.0)
    except Exception:
        # Fallback to global EI
        z = (f_best - mu) / sigma_safe
        ei_local = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei_local = np.maximum(ei_local, 0.0)
    
    # Also compute global EI as a secondary signal
    z_global = (f_best - mu) / sigma_safe
    ei_global = (f_best - mu) * norm.cdf(z_global) + sigma_safe * norm.pdf(z_global)
    ei_global = np.maximum(ei_global, 0.0)
    
    # Combine local and global EI
    # Weight more towards local EI to escape local minima
    # Use a mixing parameter that adapts based on uncertainty
    sigma_mean = sigma_safe.mean()
    # When sigma is high, trust local exploration more
    mix = np.clip(sigma_safe / (sigma_mean + 1e-8), 0.0, 1.0)
    
    ei_combined = 0.6 * ei_local + 0.4 * ei_global
    
    # Cost-aware scoring
    # The cost is 1 + mean(x_scaled), so it ranges from 1 to 2
    # The minimum of ST2D is at (-2.9, -2.9), which in [0,1]^2 is at (0.21, 0.21)
    # so cost = 1 + 0.21 = 1.21
    # Use a cost penalty that rewards low-cost evaluations
    cost_penalty = np.power(cost, 1.3)
    
    scores = ei_combined / cost_penalty
    
    # Add a small exploration bonus for high uncertainty regions
    # to help escape local minima
    exploration_bonus = sigma_safe / (sigma_mean + 1e-8)
    scores = scores + 0.05 * exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return simple fallback
        return np.where(np.isfinite(ei_global), ei_global, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
