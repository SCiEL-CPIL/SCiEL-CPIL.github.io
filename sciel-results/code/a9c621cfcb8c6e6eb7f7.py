"""Acquisition function for Bayesian optimization of beale 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Deterministic surrogate-guided heuristic combining exploitation and exploration.
    
    Uses a composite score: score = -mu + gamma * sigma
    where gamma is dynamically adjusted based on the current state of optimization.
    
    mu: GP mean predictions (2000,)
    sigma: GP standard deviation predictions (2000,)
    f_best: best observed value so far
    """
    # Track elapsed time for safety margin
    t_start = time.time()
    deadline = 320.0  # 10s safety margin before 330s kill
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    if mu.size != 2000:
        raise ValueError("mu must have shape (2000,)")
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute basic statistics
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    
    # Dynamic gamma based on optimization state
    # When f_best is close to 0 (good), we want more exploitation (lower gamma)
    # When f_best is far from 0 (bad), we want more exploration (higher gamma)
    
    # Normalize f_best relative to the range of mu
    mu_range = mu_max - mu_min
    if mu_range < 1e-12:
        mu_range = 1.0
    
    # f_best relative position: 0 means at best mu, 1 means at worst mu
    if f_best <= mu_min:
        f_best_norm = 0.0
    elif f_best >= mu_max:
        f_best_norm = 1.0
    else:
        f_best_norm = (f_best - mu_min) / mu_range
    
    # Base gamma: higher when f_best is poor (high f_best_norm)
    # gamma ranges from ~0.5 (exploit) to ~3.0 (explore)
    base_gamma = 0.5 + 2.5 * f_best_norm
    
    # Add uncertainty-based adjustment: if sigma is high overall, explore more
    sigma_factor = 1.0 + 0.5 * (sigma_mean / (abs(mu_min) + 1e-9))
    gamma = base_gamma * sigma_factor
    
    # Clamp gamma to reasonable range
    gamma = np.clip(gamma, 0.3, 5.0)
    
    # Compute composite score
    # Higher score = better (we want to maximize this)
    # -mu favors low predicted values (exploitation)
    # gamma * sigma favors high uncertainty (exploration)
    score = -mu + gamma * sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check time budget
    elapsed = time.time() - t_start
    if elapsed > deadline:
        # Return simple exploitation score as fallback
        score = -mu
        score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
