"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track elapsed time for deadline guard
_start_time = time.time()
_TIMEOUT = 320.0  # Safety margin for 330s limit

def acquisition(mu, sigma, f_best):
    """
    Pareto-frontier guided exploration acquisition function.
    
    Strategy:
    1. Identify the "exploitation anchor": the candidate with the lowest predicted mean (mu).
       This represents the current best estimate of the minimum value.
    2. Compute the normalized distance from each candidate to this anchor in prediction space.
       We use the difference in mu values, scaled by the typical scale of mu.
    3. Score = sigma * normalized_distance.
       This prioritizes points that are both uncertain (high sigma) and far from the current
       best prediction (high normalized distance), encouraging exploration of distant,
       unexplored regions of the search space.
       
    Higher scores mean evaluate next.
    """
    # Check deadline
    if time.time() - _start_time > _TIMEOUT:
        # Return a safe fallback: pure exploitation (lowest mu)
        # Score = -mu, so lowest mu gets highest score
        return -mu

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Step 1: Find the exploitation anchor (lowest mu)
    anchor_idx = np.argmin(mu)
    anchor_mu = mu[anchor_idx]
    
    # Step 2: Compute normalized distance from anchor in prediction space
    # Distance = mu - anchor_mu (positive for points with higher predicted value)
    # We want to reward points that are FAR from the anchor, so we use the absolute difference
    # but scaled to be in a reasonable range.
    
    # Scale the distances by the standard deviation of mu to normalize
    mu_std = np.std(mu)
    if mu_std < 1e-9:
        mu_std = 1e-9
    
    # Normalized distance: how many sigma_mu away from the anchor
    # We use (mu - anchor_mu) / mu_std, but since we want to explore far from the best,
    # we use the absolute value or just the positive deviation.
    # Actually, for exploration, we want points that are NOT like the current best.
    # The anchor has mu = min(mu). Other points have mu >= anchor_mu.
    # Distance = (mu - anchor_mu) / mu_std
    
    normalized_dist = (mu - anchor_mu) / mu_std
    
    # Step 3: Score = sigma * normalized_dist
    # This rewards high uncertainty AND distance from the current best prediction.
    # Points near the anchor (low normalized_dist) get low scores even if sigma is high.
    # Points far from the anchor (high normalized_dist) get high scores if sigma is high.
    
    scores = sigma_safe * normalized_dist
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
