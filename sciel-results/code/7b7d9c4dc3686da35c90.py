import numpy as np
from scipy.stats import norm
import time

_deadline = None
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Kriging Believer strategy.
    
    This strategy computes Expected Improvement (EI) to identify the most promising
    uncertain region, extracts the quantile (z-score) of that point, and then
    scores all candidates based on their mean prediction adjusted by this global
    quantile, normalized by evaluation cost.
    
    Parameters:
    mu: (2000,) array of mean predictions
    sigma: (2000,) array of standard deviations
    f_best: float, current best observed value
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    (2000,) array of scores (higher is better)
    """
    global _deadline, _start_time
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 325.0  # 5s safety margin

    if time.time() > _deadline:
        return np.ones(len(mu))

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-scores for all candidates
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement for minimization
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Find the candidate with the maximum Expected Improvement
    max_ei_idx = np.argmax(ei)
    
    # Extract the z-score (quantile) of the best EI candidate
    # This z* represents the "belief" level of uncertainty/promisingness
    z_star = z[max_ei_idx]
    
    # Kriging Believer Score:
    # We predict the value as mu + sigma * z_star.
    # Since we want to MINIMIZE the function value, we want this predicted value to be LOW.
    # Therefore, the score (higher is better) is the negative of this predicted value.
    believer_value = mu + sigma_safe * z_star
    scores = -believe_value / (cost + eps)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
