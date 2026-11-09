"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending EI and UCB.
    
    Parameters:
    -----------
    mu : array, shape (2000,)
        GP mean predictions
    sigma : array, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : array, shape (2000,)
        Evaluation costs for each candidate
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Cost normalization: normalize costs to [0, 1] range for weighting
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-12:
        cost_norm = np.ones_like(cost)
    else:
        cost_norm = (cost - cost_min) / cost_range
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # EI is positive when mu < f_best (potential improvement)
    ei = np.maximum(ei, 0.0)
    
    # Upper Confidence Bound (UCB) for exploration
    # Use a beta parameter that balances exploitation and exploration
    # Higher beta = more exploration
    beta = 0.5
    ucb = mu + beta * sigma
    
    # For minimization, we want low mu and high uncertainty
    # So we negate UCB to make it an "improvement" score
    ucb_improve = -(ucb)
    
    # Cost-aware normalization
    # Penalize expensive points more heavily
    # Use exponential cost penalty for stronger discrimination
    cost_penalty = np.exp(-2.0 * cost_norm)
    
    # Blend EI and UCB with adaptive weight
    # When uncertainty is high, weight UCB more; when low, weight EI more
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    if sigma_max < 1e-12:
        uncertainty_level = 0.0
    else:
        uncertainty_level = sigma_mean / sigma_max
    
    # Adaptive blend: more UCB when uncertainty is high
    # weight_ei goes from 0.3 (high uncertainty) to 0.8 (low uncertainty)
    weight_ei = 0.3 + 0.5 * (1.0 - uncertainty_level)
    weight_ucb = 1.0 - weight_ei
    
    # Combine EI and UCB scores
    # Normalize each to have similar scale
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    ucb_max = np.max(np.abs(ucb_improve)) if np.max(np.abs(ucb_improve)) > 1e-12 else 1.0
    
    ei_norm = ei / ei_max
    ucb_norm = ucb_improve / ucb_max
    
    # Blend
    blended = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Apply cost penalty
    scores = blended * cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Add a tiny bit of exploration noise to break ties and encourage diversity
    # This is deterministic based on index to be reproducible
    idx = np.arange(len(scores))
    noise = 1e-6 * np.sin(idx * 0.1)
    scores = scores + noise
    
    return scores
# EVOLVE-BLOCK-END
