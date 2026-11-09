"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.stats import rankdata


def acquisition(mu, sigma, f_best):
    """
    Rank-based UCB acquisition function.
    Uses Upper Confidence Bound with a fixed beta, then converts scores to ranks.
    This structural change makes the selection invariant to the scale of the
    posterior predictive distribution, which can improve robustness in
    multimodal landscapes like Michalewicz 2D.
    """
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB with a fixed beta
    # Beta = 2.0 is a standard choice for balancing exploration and exploitation
    beta = 2.0
    ucb_scores = -mu + beta * sigma_safe
    
    # Convert to ranks
    # Higher UCB score -> Higher rank -> Higher acquisition score
    # rankdata returns 1..N, so we normalize to [0, 1]
    ranks = rankdata(ucb_scores, method='average')
    scores = ranks / len(ranks)
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
