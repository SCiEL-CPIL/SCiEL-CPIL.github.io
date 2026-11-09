"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.stats import rankdata
import time

# Track call count to adaptively tune exploration
_call_count = [0]


def acquisition(mu, sigma, f_best):
    """
    Adaptive Rank-based Hybrid Acquisition Function.
    Combines Expected Improvement (EI) and Upper Confidence Bound (UCB) scores.
    Uses adaptive beta that decreases over time to balance exploration/exploitation.
    """
    _call_count[0] += 1
    n_calls = _call_count[0]
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement component
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Adaptive beta: start with high exploration, decrease over time
    # Total budget is 30 evaluations, so we can adapt over that range
    progress = min(n_calls / 30.0, 1.0)
    beta = 2.0 * (1.0 - 0.5 * progress)  # Beta ranges from 2.0 to 1.0
    
    # UCB component: linear in uncertainty
    ucb = -mu + beta * sigma_safe
    
    # Convert to ranks (higher rank = higher score)
    ei_ranks = rankdata(ei)
    ucb_ranks = rankdata(ucb)
    
    # Normalize ranks to [0, 1] for stable combination
    ei_norm = ei_ranks / len(ei_ranks)
    ucb_norm = ucb_ranks / len(ucb_ranks)
    
    # Adaptive weights: more exploration early, more exploitation later
    w_ei = 0.5 + 0.2 * progress  # 0.5 to 0.7
    w_ucb = 1.0 - w_ei           # 0.5 to 0.3
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
