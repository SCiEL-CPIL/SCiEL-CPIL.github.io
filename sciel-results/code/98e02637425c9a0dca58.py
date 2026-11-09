import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Expected Improvement (CAEI).
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, the best (lowest) function value observed so far.
        cost: (2000,) array of evaluation costs for each candidate.
        
    Returns:
        (2000,) array of scores (higher is better).
    """
    # Ensure sigma is positive to avoid division by zero issues in Z calculation
    # If sigma is 0, we handle it separately or with a small epsilon.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Z score for Expected Improvement
    # Z = (mu - f_best) / sigma
    Z = (mu - f_best) / sigma_safe
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(-Z) + sigma * phi(Z)
    # Where Phi is the CDF of the standard normal distribution
    # And phi is the PDF of the standard normal distribution
    
    # Phi(-Z) is the probability that f(x) < f_best
    # Since we are minimizing, we want f(x) to be lower than f_best.
    # The improvement is max(0, f_best - f(x)).
    # E[max(0, f_best - f(x))] = (f_best - mu) * P(f(x) < f_best) + sigma * phi(Z)
    
    phi_Z = norm.pdf(Z)
    Phi_neg_Z = norm.cdf(-Z)
    
    EI = (f_best - mu) * Phi_neg_Z + sigma_safe * phi_Z
    
    # Cost-Adjusted EI
    # We divide by cost to prioritize cheaper evaluations
    # Ensure cost is not zero (it's >= 1 per problem description)
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = EI / cost_safe
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
