import numpy as np
from scipy.stats import norm
import time

# Global state for tracking evaluated points (approximated via mu/sigma patterns)
# Since we don't get coordinates, we use a simple diversity penalty based on
# the structure of the GP posterior to encourage exploration of new regions.

def acquisition(mu, sigma, f_best):
    """
    Diversity-weighted Expected Improvement for Bayesian optimization.
    
    The key insight is that for multimodal functions like Styblinski-Tang,
    standard EI tends to exploit a single local minimum. We combine EI with
    a diversity bonus that rewards points where the GP is uncertain (high sigma)
    relative to the local neighborhood, encouraging exploration of new basins.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill
    
    # Handle division by zero or very small sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement
    target = f_best
    z_ei = (target - mu) / sigma_safe
    ei = (target - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    ei = np.maximum(ei, 0)
    
    # Compute UCB (Upper Confidence Bound) for exploration
    # The beta parameter controls the exploration-exploitation tradeoff
    # We use a moderately large beta to encourage exploration
    beta = 0.5
    ucb = -mu + beta * sigma_safe
    
    # Normalize EI to [0, 1]
    ei_max = np.max(ei)
    if ei_max > 1e-9:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Normalize UCB to [0, 1]
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    if ucb_max - ucb_min > 1e-9:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Diversity-aware weighting:
    # We want to boost points that are far from the current "exploited" region.
    # Since we don't have coordinates, we use the GP's uncertainty structure.
    # Points with high sigma relative to their neighbors are likely in unexplored regions.
    
    # Compute a local density proxy using the sigma values themselves.
    # In a well-explored region, sigma tends to be low and smooth.
    # In unexplored regions, sigma can be high and variable.
    
    # Use a simple approach: weight by sigma to favor uncertain regions
    # But also penalize regions where EI is already high (exploited)
    
    # Combine EI and UCB with adaptive weighting
    # Use a sigmoid-like blend that favors exploration more when EI is low
    ei_weight = np.clip(ei_norm * 0.8, 0.0, 0.8)
    ucb_weight = 1.0 - ei_weight
    
    score = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # Add a small bonus for high uncertainty to further encourage exploration
    sigma_norm = sigma_safe / (np.max(sigma_safe) + 1e-9)
    score += 0.1 * sigma_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return score
    
    return score
