"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.stats import rankdata


def acquisition(mu, sigma, f_best):
    """
    Rank-based Hybrid Acquisition Function.
    Combines Expected Improvement (EI) and Upper Confidence Bound (UCB) scores.
    Both scores are converted to ranks before combining to ensure scale invariance.
    """
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement component
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # UCB component: linear in uncertainty
    # Use a fixed beta for UCB to avoid adaptive tuning issues
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # Convert to ranks (higher rank = higher score)
    # rankdata returns ranks from 1 to N, so we reverse them to get higher rank for higher score
    # Actually, rankdata assigns 1 to the smallest value. We want higher score to have higher rank.
    # So we use rankdata(1 - value) or just use rankdata and then flip?
    # Let's use rankdata on the scores themselves, where higher score should get higher rank.
    # rankdata([1, 2, 3]) -> [1, 2, 3]. So higher value gets higher rank. This is what we want.
    ei_ranks = rankdata(ei)
    ucb_ranks = rankdata(ucb)
    
    # Normalize ranks to [0, 1] for stable combination
    ei_norm = ei_ranks / len(ei_ranks)
    ucb_norm = ucb_ranks / len(ucb_ranks)
    
    # Weighted combination: 
    # Use a fixed weight that balances exploration and exploitation.
    # For Michalewicz 2D, a bit more exploration is often needed to find all modes.
    # Let's try 0.5 EI and 0.5 UCB, or slightly more EI since it's more targeted.
    w_ei = 0.6
    w_ucb = 0.4
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
