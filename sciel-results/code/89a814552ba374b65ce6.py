"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 325.0  # Safety margin before the 330s kill limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses a hybrid of Expected Improvement and a diversity-promoting term
    based on the variance of the GP posterior to encourage exploration.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    # Check for timeout
    if time.time() - _start_time > _TIMEOUT:
        return np.zeros_like(mu)
    
    # Ensure sigma is non-negative and finite, avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Expected Improvement component
    z = (f_best - mu) / sigma
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Diversity/exploration component:
    # We want to encourage points where the GP is uncertain (high sigma).
    # But we don't want to just maximize sigma, as that could lead to
    # exploring far away from the current best. Instead, we use a
    # term that balances exploitation (EI) and exploration (sigma).
    #
    # A common approach is to use UCB: mu + beta * sigma
    # But for minimization, we want to minimize, so we use:
    # score = EI + alpha * sigma
    #
    # However, EI already incorporates sigma. Let's think differently.
    #
    # Another approach: use a normalized combination.
    # The EI scores can vary in scale. Let's normalize them.
    #
    # Actually, let's use a different strategy: 
    # The key insight for high-dimensional optimization is that we need
    # to balance local refinement with global exploration. 
    #
    # Let's use a score that is:
    # 1. Primarily based on EI (exploitation)
    # 2. Plus a bonus for high uncertainty (exploration)
    # 3. But with a cap to prevent pure exploration
    #
    # score = EI + beta * min(sigma, sigma_max)
    # where sigma_max is a reasonable upper bound on sigma
    #
    # But what is a reasonable sigma_max? We can use the median or a
    # percentile of sigma as a reference.
    
    # Compute a reference sigma scale
    sigma_ref = np.median(sigma)
    
    # Exploration bonus: scaled by how much more uncertain this point is
    # compared to the median uncertainty
    exploration_bonus = sigma_ref * (sigma / sigma_ref - 1.0)
    exploration_bonus = np.maximum(exploration_bonus, 0.0)
    
    # Scale the exploration bonus to be comparable to EI
    # EI has units of the objective, sigma also has units of the objective
    # Let's use a small coefficient to not overwhelm EI
    beta = 0.1
    
    scores = ei + beta * exploration_bonus
    
    # Clean up non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
