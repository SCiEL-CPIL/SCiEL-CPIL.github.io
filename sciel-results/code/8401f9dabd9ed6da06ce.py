import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Probability of Improvement (PoI) divided by cost.
    
    PoI focuses on the likelihood of improving over f_best rather than the
    expected magnitude of improvement. This is more stable when sigma varies
    widely across candidates. Dividing by cost naturally balances exploration
    (high sigma -> higher chance of improvement) vs exploitation (low mu ->
    higher improvement probability) while penalizing expensive evaluations.
    
    Parameters:
    mu : array-like, shape (N,)
        Mean of the GP posterior in objective units.
    sigma : array-like, shape (N,)
        Standard deviation of the GP posterior in objective units.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (N,)
        Cost of evaluating each candidate point.
        
    Returns:
    scores : array, shape (N,)
        Acquisition scores (higher is better).
    """
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Probability of Improvement: P(f(x) < f_best)
    # For f(x) ~ N(mu, sigma^2):
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Clip to avoid exactly 0 or 1 which can cause issues
    poi = np.clip(poi, eps, 1.0 - eps)
    
    # Cost-aware: divide by cost to penalize expensive points
    score = poi / cost_safe
    
    # Handle non-finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
