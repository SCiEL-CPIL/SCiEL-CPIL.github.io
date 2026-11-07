# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 300.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware Expected Improvement (CA-EI) with Analytic Gaussian Integration.
    
    Strategy:
    1. Use Expected Improvement (EI) instead of UCB. EI is theoretically grounded
       in the probability of improvement and is more stable in the presence of 
       noise and multimodal landscapes.
    2. Cost-aware adjustment: Divide the EI by the cost to favor cheaper evaluations.
       This is equivalent to maximizing the expected improvement per unit cost.
    3. The EI formula for minimization is:
       EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
       where z = (f_best - mu) / sigma
       Phi is the standard normal CDF and phi is the standard normal PDF.
       
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Predictive mean from the GP.
    sigma : array-like, shape (2000,)
        Predictive standard deviation from the GP.
    f_best : float
        Lowest observed function value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Deadline guard: Return zeros if we are close to the time limit
    if time.time() - _START_TIME > _TIME_LIMIT - 5.0:
        return np.zeros_like(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Calculate Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi(z) is the CDF of the standard normal distribution
    # phi(z) is the PDF of the standard normal distribution
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware adjustment: Divide by cost to penalize expensive evaluations
    # This favors points with lower cost (center of the domain)
    scores = ei / cost
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
