"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Predictive Entropy Search (PES) variant acquisition function.
    
    We approximate the mutual information I(f(x); m*) where m* is the global
    minimum, using a practical surrogate that balances:
    1. The probability that x improves on f_best (exploration of promising regions)
    2. The expected improvement magnitude (exploitation)
    3. A penalty for high uncertainty that doesn't lead to improvement
    
    The score is designed to favor points where observing f(x) would most likely
    change our estimate of the global minimum, which is the core idea of PES.
    
    We use: score = P(improve) * (expected_improvement / (sigma + eps))
    This normalizes the expected improvement by local uncertainty, rewarding
    points with high-confidence improvements over noisy regions.
    """
    eps = 1e-9
    sigma_safe = sigma + eps
    z = (f_best - mu) / sigma_safe
    
    # Probability that f(x) < f_best
    p_improve = norm.cdf(z)
    
    # Expected improvement: E[max(0, f_best - f(x))]
    ei = (f_best - mu) * p_improve + sigma * norm.pdf(z)
    
    # PES-like score: normalize EI by local uncertainty
    # This rewards points where the expected improvement is large relative to noise
    score = p_improve * (ei / sigma_safe)
    
    # Add a small exploration bonus for points with high uncertainty that
    # have a reasonable chance of improvement
    uncertainty_bonus = sigma_safe * p_improve * (1 - p_improve)
    
    # Combine: primary PES score + secondary exploration term
    final_score = score + 0.1 * uncertainty_bonus
    
    # Ensure finite values
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    return final_score
# EVOLVE-BLOCK-END
