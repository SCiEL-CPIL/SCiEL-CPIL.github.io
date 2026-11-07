import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a cost-aware exploration bonus.
    
    The primary signal is EI, which drives exploitation towards the minimum.
    A secondary signal is a cost-modulated uncertainty bonus (UCB-like)
    that encourages exploration in regions with high uncertainty, but
    discounts this exploration by cost to avoid wasting budget on
    expensive, uncertain regions.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # Since we are minimizing, improvement is f_best - mu (positive if mu < f_best)
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = diff * cdf_z + sigma_safe * pdf_z
    
    # Cost-aware exploration bonus
    # We want to explore high-uncertainty areas, but less so if they are expensive.
    # cost is in [1, 2].
    # Let's define a multiplier that is 1.0 for cost=1.0 and 0.5 for cost=2.0.
    # multiplier = 1.5 - 0.5 * cost
    # If cost=1, mult=1.0. If cost=2, mult=0.5.
    multiplier = 1.5 - 0.5 * cost
    multiplier = np.clip(multiplier, 0.1, 1.0)
    
    # Exploration coefficient
    # A small beta to not overpower the EI signal, but provide a tie-breaker
    # and encourage moving away from the current best if uncertainty is high.
    beta = 0.5
    
    # The bonus term: beta * multiplier * sigma
    # This adds a positive score for high sigma, scaled down by cost.
    bonus = beta * multiplier * sigma_safe
    
    # Total score
    # Higher is better.
    # EI is higher for better points.
    # Bonus is higher for more uncertain (and cheaper) points.
    score = ei + bonus
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    # EI should be non-negative, but the bonus makes the whole thing positive.
    # If mu is very good, EI is large. If mu is bad, EI is ~0, but bonus might be non-zero.
    # We want to ensure we don't pick points that are clearly bad (mu >> f_best) 
    # unless the uncertainty is huge.
    # The current formulation does this naturally: EI -> 0, bonus is bounded by beta*sigma.
    
    return score
