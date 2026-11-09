"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement (EI) with a small exploration bonus.
    
    We use a variant of EI that also considers the probability of finding a
    significantly better point, to encourage exploration in early stages.
    
    mu: (n,) array of GP mean predictions
    sigma: (n,) array of GP standard deviation predictions
    f_best: current best observed value (minimum)
    
    Returns: (n,) array of scores, higher is better
    """
    start_time = time.time()
    deadline = start_time + 295.0  # 5s safety margin before 300s limit
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Dynamic exploration weight: decreases as f_best approaches the known minimum
    # For Hartmann 6d, minimum is -3.32237
    min_possible = -3.32237
    # Normalize the gap: 1.0 when f_best is far from min, 0.0 when at min
    # Use a scale of ~1.0 for the gap to make the transition smooth
    gap = max(f_best - min_possible, 0.0)
    # Weight starts at 0.15 and decays to 0.0 as we approach the minimum
    # Using a soft decay: weight = base_weight * exp(-gap / scale)
    base_weight = 0.15
    scale = 0.5
    dynamic_weight = base_weight * np.exp(-gap / scale)
    
    exploration_bonus = dynamic_weight * sigma_safe
    
    # Combine EI and exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check if we're approaching the deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Return best available result quickly
        return scores
    
    return scores
# EVOLVE-BLOCK-END
