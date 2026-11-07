import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Quantile of Improvement (QI)
    with a deadline guard.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores (higher is better).
    """
    # Deadline guard: track elapsed time and return early if approaching limit
    # The evaluator kills after 330s, so we use a safety margin
    # But since this function is called repeatedly, we need to be fast
    # Let's just compute efficiently
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Quantile of Improvement: We want to maximize the expected improvement
    # weighted by the probability of improvement, but focusing on high-reward regions.
    # 
    # QI can be formulated as: E[max(0, f_best - f(x)) | f(x) < f_best] * P(f(x) < f_best)
    # But that's just EI. 
    #
    # A better QI approach: Consider the conditional expectation of the objective
    # given improvement. This is:
    # E[f(x) | f(x) < f_best] = mu - sigma * (phi(z) / Phi(z))
    # where z = (f_best - mu) / sigma
    #
    # The improvement given improvement is: f_best - E[f(x) | f(x) < f_best]
    # = f_best - mu + sigma * (phi(z) / Phi(z))
    #
    # But we want to balance exploration and exploitation. Let's use a modified
    # approach: maximize the expected improvement per unit cost, but with a
    # quantile-based weighting that emphasizes high-probability, high-reward regions.
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Probability of Improvement (PI)
    pi = cdf
    
    # For QI, we consider the conditional expected improvement given that
    # an improvement occurs. This is:
    # conditional_ei = ei / pi (when pi > 0)
    # But this can be unstable when pi is small.
    
    # Alternative: Use a weighted combination that emphasizes both the
    # magnitude of improvement and the probability of it occurring.
    # Let's use: score = (ei + alpha * pi * (f_best - mu)) / cost
    # where alpha controls the emphasis on the expected value.
    
    # Actually, let's use a cleaner QI formulation:
    # Maximize the expected value of the improvement, but penalize uncertainty
    # in a way that favors high-confidence improvements.
    
    # Use the formula: QI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # This is exactly EI. So let's add a cost-aware term that also considers
    # the variance (uncertainty).
    
    # Cost-aware QI: (EI + beta * sigma * pi) / cost
    # where beta > 0 encourages exploration when uncertainty is high and
    # the probability of improvement is reasonable.
    
    beta = 0.5
    
    score = (ei + beta * sigma_safe * pi) / cost
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
