"""Cost-aware acquisition for Bayesian optimization on powell 4d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level state to track time budget
_start_time = time.time()
_TIME_LIMIT = 290.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled_to_01))
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values (higher = more desirable)
    """
    # Check time budget
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Return a simple fallback if we're running out of time
        return np.zeros_like(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: EI per unit cost
    # But also incorporate a dynamic penalty that increases as we get closer to the minimum
    # The idea: as f_best approaches 0, we should be more aggressive about cost
    
    # Progress measure: how close are we to the theoretical minimum (0)?
    # f_best starts around 10-20 for powell 4d and decreases toward 0
    # Use a soft progress indicator
    progress = np.clip(1.0 - f_best / 20.0, 0.0, 1.0)
    
    # Dynamic cost penalty: as progress increases, penalize cost more heavily
    # This shifts from exploration (early, high f_best) to exploitation (late, low f_best)
    cost_penalty_weight = 0.5 + 1.5 * progress
    
    # Normalize cost to [0, 1] range for better scaling
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost)
    
    # Score: EI minus a cost penalty that scales with progress
    # This encourages finding improvements that are "cheap" relative to their benefit
    scores = ei - cost_penalty_weight * cost_norm * np.mean(ei)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
