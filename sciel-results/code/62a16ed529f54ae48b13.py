import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) with a cost-aware scaling that
    rewards points with high expected improvement relative to their cost.
    
    Parameters:
    mu: (2000,) array of GP means
    sigma: (2000,) array of GP standard deviations
    f_best: float, the lowest observed value
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    (2000,) array of acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate Z-scores for Expected Improvement (Minimization)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware scaling:
    # Cost ranges from 1 to 2 (1 + mean(x scaled to [0,1]))
    # We use a cost-aware metric that divides EI by a cost-related factor.
    # This effectively normalizes the improvement by the "price" of the evaluation.
    
    # A key insight: points near the boundary (cost=1) are cheaper.
    # We want to strongly prefer cheap points when EI is similar.
    
    # Strategy: Use a power-law cost penalty that is smooth and differentiable.
    # cost_penalty = (1 / cost)^alpha
    # For alpha=1, this is 1/cost. For alpha=2, it's 1/cost^2.
    
    # Let's try a combination: EI / cost^alpha with alpha tuned for this problem.
    # Since cost is in [1, 2], 1/cost ranges from 0.5 to 1.0.
    # 1/cost^2 ranges from 0.25 to 1.0.
    
    # A more nuanced approach: use the cost to scale the exploration-exploitation tradeoff.
    # High cost should require higher EI to be selected.
    
    # Let's use: score = EI / (cost^alpha) where alpha is chosen to balance
    # the cost range effectively.
    
    alpha = 1.5
    cost_penalty = 1.0 / (cost ** alpha)
    
    score = ei * cost_penalty
    
    # Additionally, add a small exploration bonus for points with high uncertainty
    # that are cheap, to ensure we don't get stuck in exploitation.
    # But EI already captures this somewhat.
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
