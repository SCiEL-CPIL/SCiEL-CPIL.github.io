"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (POI) acquisition function for Bayesian optimization.
    
    Strategy:
    - POI provides a softer, rank-based exploration pressure compared to UCB's linear sigma weighting.
    - It computes the probability that a candidate point improves upon the current best (f_best).
    - For minimization: P(f(x) < f_best) = Phi((f_best - mu) / sigma).
    - This approach may better balance exploitation of known low regions and exploration of high-variance areas in multimodal functions like Branin.
    - The CDF transformation naturally dampens the influence of extreme sigma values compared to UCB, potentially preventing over-exploration in distant regions while still maintaining diversity.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for POI
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (POI) for minimization
    # P(f(x) < f_best) = Phi(z) where z = (f_best - mu) / sigma
    poi_scores = norm.cdf(z)
    
    # Ensure finite scores
    poi_scores = np.where(np.isfinite(poi_scores), poi_scores, 0.0)
    
    return poi_scores
# EVOLVE-BLOCK-END
