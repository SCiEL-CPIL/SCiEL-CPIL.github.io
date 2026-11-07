# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-entropy heuristic: maximize expected information gain about the global
    minimum per unit cost.
    
    The mutual information between a candidate's objective value and the posterior
    over the global minimum is approximated by the expected reduction in entropy
    of the global minimum distribution. A common proxy is the Expected Improvement
    (EI) which measures how much we expect to improve upon f_best, but weighted
    by uncertainty (sigma). However, to capture 'information gain' more directly,
    we use a variant that rewards high uncertainty (high sigma) in regions where
    improvement is likely, while penalizing cost.
    
    Specifically, we use:
      score = (EI + sigma * uncertainty_bonus) / (cost + epsilon)
    
    But a more principled approach for information gain is to use the Expected
    Value of Information (EVI), which for a single observation is related to the
    reduction in posterior variance. A practical surrogate is to combine EI with
    a term that rewards high sigma (uncertainty) in promising regions.
    
    We use: score = (f_best - mu) * Phi(z) + sigma * phi(z)  (this is EI)
    Then we modify it to account for information gain by adding a term proportional
    to sigma in regions where sigma is high and mu is low (promising and uncertain).
    
    Actually, let's stick with a well-known cost-aware acquisition: 
    Expected Improvement divided by cost, but with a small addition to handle
    the information gain aspect. The key insight is that points with high sigma
    in low-mu regions provide more information about the true minimum.
    
    We'll use: score = EI / cost + beta * sigma / cost
    where beta is a small constant to encourage exploration of uncertain regions.
    """
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Information gain proxy: reward high uncertainty (sigma) in promising regions
    # Use a soft weighting: sigma * exp(-mu / scale) to favor low-mu, high-sigma points
    # But to keep it simple and stable, we'll add a small multiple of sigma
    info_gain = 0.1 * sigma_safe
    
    # Total score: (EI + info_gain) / cost
    score = (ei + info_gain) / (cost + eps)
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
