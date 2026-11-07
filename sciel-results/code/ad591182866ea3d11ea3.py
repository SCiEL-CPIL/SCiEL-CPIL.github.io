# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = time.time() + 320  # 320s wall-clock budget (10s safety margin before 330s kill)


def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Knowledge Gradient with sublinear cost discount.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,) - GP mean predictions
    sigma : ndarray, shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : ndarray, shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Check deadline early
    if _check_deadline():
        # Return a safe fallback: just use negative cost (prefer cheap points)
        return -cost.copy()
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Knowledge Gradient approximation:
    # KG(x) = E[max(f_best - f(x), 0)] - E[max(f_best - f*(x), 0)]
    # where f*(x) is the best value found after evaluating at x and then optimally
    # continuing the search. A common approximation uses:
    # KG(x) ≈ E[max(f_best - f(x), 0)] - (1 - CDF) * (f_best - mu) - sigma * PDF
    # But a simpler and effective approach for cost-aware BO is to use
    # the probability-weighted improvement with sublinear cost discount.
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected improvement (EI) component
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability that this point improves on f_best
    prob_improve = norm.cdf(z)
    
    # Knowledge Gradient approximation:
    # The key insight is that KG values points where there's a high probability
    # of finding a new minimum, not just expected improvement.
    # A practical approximation: weight EI by the probability of improvement
    # and also consider the "surprise" factor (how much better than current best)
    
    # Sublinear cost discount: 1/sqrt(cost) instead of 1/cost
    # This favors expensive points more than linear cost discount,
    # allowing exploration of high-value but costly regions
    cost_discount = 1.0 / np.sqrt(np.maximum(cost, 1e-9))
    
    # Combine: use EI weighted by probability of improvement, discounted by sqrt(cost)
    # This balances:
    # 1. Expected improvement (exploitation)
    # 2. Probability of being a new minimum (exploration via KG-like behavior)
    # 3. Sublinear cost penalty (allows some expensive evaluations)
    
    # Additional term: reward high uncertainty regions that could contain new minima
    # This is a proxy for the "option value" in KG
    uncertainty_bonus = sigma_safe * prob_improve
    
    # Final score: cost-discounted combination of EI and uncertainty bonus
    scores = (ei + 0.5 * uncertainty_bonus) * cost_discount
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
