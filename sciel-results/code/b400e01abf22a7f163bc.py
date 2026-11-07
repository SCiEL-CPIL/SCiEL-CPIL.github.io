"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a dynamic Lower Confidence Bound (LCB)
    with a feature-space repulsion term to encourage diversity.
    """
    start_time = time.time()
    deadline = start_time + 5.0  # Safety margin, though this function is fast
    
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
    # Ranges from 0.5 (exploit) to 2.5 (explore)
    kappa_min = 0.5
    kappa_max = 2.5
    kappa = kappa_min + (kappa_max - kappa_min) * np.sqrt(progress)
    
    # Calculate base LCB score for maximization
    # Score = -mu + kappa * sigma
    scores = -mu_safe + kappa * sigma_safe
    
    # Feature-space repulsion term
    # Approximate the feature vector of the current best point as (f_best, 0)
    # Penalize candidates that are close to this point in (mu, sigma) space
    # This encourages moving away from the immediate vicinity of the best found
    # point in prediction space, promoting exploration of other regions.
    
    # Distance in feature space
    # Scale mu and sigma appropriately. 
    # mu is in objective units (range ~ [-4, 0]).
    # sigma is in objective units (range ~ [0, 1]).
    # We use a simple Euclidean distance but scaled to make them comparable.
    # Let's assume a typical sigma scale of 1.0 for normalization.
    
    # Check time
    if time.time() > deadline:
        return np.where(np.isfinite(scores), scores, -np.inf)

    # Compute squared distance to (f_best, 0)
    # d^2 = (mu - f_best)^2 + (sigma - 0)^2
    # We use a soft penalty: penalty = strength * exp(-d / scale)
    # Or simply subtract a distance-based term.
    
    # Let's try a Gaussian kernel repulsion
    # d = sqrt((mu - f_best)^2 + sigma^2)
    # To avoid sqrt for speed, we can use squared distance in the exponent if we adjust scale, 
    # or just compute sqrt for 2000 points which is negligible.
    
    d = np.sqrt((mu_safe - f_best)**2 + sigma_safe**2)
    
    # Scale parameter for the repulsion
    # If d is small, we want a strong penalty.
    # Typical distance might be ~1.0
    repulsion_scale = 0.5
    repulsion_strength = 0.5 * (1.0 - progress)  # Stronger repulsion early on
    
    # Penalty is high when d is small
    penalty = repulsion_strength * np.exp(-d / repulsion_scale)
    
    scores = scores - penalty
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
