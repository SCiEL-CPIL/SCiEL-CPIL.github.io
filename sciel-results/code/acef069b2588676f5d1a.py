"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending Expected Improvement (EI) and UCB.
    
    Args:
    mu: (2000,) array of GP means
    sigma: (2000,) array of GP standard deviations
    f_best: float, lowest observed value
    cost: (2000,) array of evaluation costs
    
    Returns:
    (2000,) array of scores (higher is better)
    """
    # Check deadline
    if time.time() > _DEADLINE - _MARGIN:
        # Return a safe, deterministic score (e.g., based on cost only) to avoid timeout
        # Minimizing cost is a valid fallback
        return 1.0 / (cost + 1e-9)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate UCB component for minimization
    # We want to maximize (beta * sigma - mu) to find low mu with high uncertainty
    # Adaptive beta: larger when f_best is high (early), smaller when f_best is low (late)
    # Using tanh to keep beta bounded
    # Scale f_best to control beta transition. 1.0 is a reasonable scale for these functions.
    beta = 1.5 * np.tanh(f_best / 1.0)
    
    ucb_score = beta * sigma - mu
    
    # Blend EI and UCB
    # Weighting: 80% EI, 20% UCB
    # This maintains the strong exploitation of EI while adding exploration
    alpha = 0.2
    combined_score = (1.0 - alpha) * ei + alpha * ucb_score
    
    # Ensure scores are finite and non-negative (though UCB can be negative)
    # If combined_score is negative, it's still valid as a relative score, 
    # but dividing by cost might amplify noise. 
    # We clip to 0 to avoid negative scores which might be confusing if the scale is mixed,
    # but actually, relative ordering is what matters.
    # However, if all scores are negative, the "best" is the least negative.
    # Let's shift the scores to be non-negative to be safe with any downstream logic that might assume positive scores.
    # min_score = np.min(combined_score)
    # combined_score = combined_score - min_score
    
    # Cost-aware scaling
    # Divide by cost. Cost is >= 1.
    scores = combined_score / cost
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
