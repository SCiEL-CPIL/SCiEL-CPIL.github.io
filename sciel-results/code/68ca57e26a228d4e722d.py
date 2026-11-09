"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()


def acquisition(mu, sigma, f_best):
    """
    Pareto-frontier guided exploration:
    1. Find the candidate with lowest predicted mean (exploitation anchor)
    2. Compute geometric distance in prediction space from this anchor
    3. Weight by sigma to prioritize high-uncertainty regions far from the best prediction
    This creates a balance between exploiting near-optimal predictions and
    exploring uncertain regions that might contain better solutions.
    """
    # Deadline guard: 330s limit. Return immediately if close to limit.
    if time.time() - _START_TIME > 320:
        return np.zeros_like(mu)
    
    # Clip sigma to avoid division by zero and ensure positive values
    sigma = np.maximum(sigma, 1e-10)
    
    # Step 1: Find the exploitation anchor - candidate with lowest predicted mean
    # This is the most promising point based on current GP predictions
    min_mu_idx = np.argmin(mu)
    anchor_mu = mu[min_mu_idx]
    anchor_sigma = sigma[min_mu_idx]
    
    # Step 2: Compute normalized distance in prediction space from the anchor
    # We use both mu and sigma dimensions to create a 2D prediction space
    # Distance captures how different each candidate is from the best prediction
    
    # Normalize mu and sigma to [0, 1] range for meaningful distance computation
    mu_range = mu.max() - mu.min() if mu.max() > mu.min() else 1.0
    sigma_range = sigma.max() - sigma.min() if sigma.max() > sigma.min() else 1.0
    
    mu_norm = (mu - mu.min()) / mu_range
    sigma_norm = (sigma - sigma.min()) / sigma_range
    
    anchor_mu_norm = (anchor_mu - mu.min()) / mu_range
    anchor_sigma_norm = (anchor_sigma - sigma.min()) / sigma_range
    
    # Euclidean distance in normalized prediction space
    dist_mu = np.abs(mu_norm - anchor_mu_norm)
    dist_sigma = np.abs(sigma_norm - anchor_sigma_norm)
    dist = np.sqrt(dist_mu**2 + dist_sigma**2)
    
    # Step 3: Weight by uncertainty (sigma) to prioritize high-uncertainty regions
    # But also incorporate the actual predicted value to avoid exploring
    # regions that are predicted to be much worse
    
    # Adaptive exploration-exploitation balance based on f_best
    # When f_best is high (early), explore more; when low (late), exploit more
    if f_best > 3.0:
        explore_weight = 0.7
        exploit_weight = 0.3
    elif f_best > 1.0:
        explore_weight = 0.5
        exploit_weight = 0.5
    else:
        explore_weight = 0.3
        exploit_weight = 0.7
    
    # Exploration component: high sigma and far from anchor in prediction space
    # Use log(sigma) to prevent sigma from dominating too much
    sigma_log = np.log1p(sigma)
    explore_score = sigma_log * dist
    
    # Exploitation component: low predicted mean (close to or below f_best)
    # Use inverse of (mu - f_best + epsilon) to reward low predictions
    exploit_score = 1.0 / (np.maximum(mu - f_best, 0.01) + 1e-6)
    
    # Normalize both components to [0, 1]
    explore_max = np.max(explore_score) if np.max(explore_score) > 0 else 1.0
    exploit_max = np.max(exploit_score) if np.max(exploit_score) > 0 else 1.0
    
    explore_norm = explore_score / explore_max
    exploit_norm = exploit_score / exploit_max
    
    # Combine with adaptive weighting
    score = explore_weight * explore_norm + exploit_weight * exploit_norm
    
    # Add a small bonus for candidates that are both low-mu AND high-sigma
    # These are the most interesting: uncertain but potentially good
    both_good = (mu < f_best) & (sigma > np.mean(sigma))
    score[both_good] += 0.1
    
    # Ensure scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
