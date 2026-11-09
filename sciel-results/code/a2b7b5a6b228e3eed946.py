import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Returns 2000 finite scores (higher = evaluate next).
    Uses Cost-Adjusted Expected Improvement (EI / Cost^0.5) to mitigate
    over-penalization of high-cost regions that may contain the global minimum.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-8)

    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement (EI)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Softened Cost-Adjusted EI
    # Using cost^0.5 instead of cost reduces the penalty for high-cost points,
    # which is crucial if the global minimum lies in a higher-cost region
    # (as in Three Hump Camel where global min cost ~1.5 > local min cost ~1.37).
    score = ei / np.sqrt(cost)
    
    # Handle any potential numerical issues
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
