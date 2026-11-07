import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Expected Improvement (EI) divided by cost.
    
    EI balances exploration (high sigma) and exploitation (low mu) by
    estimating the expected magnitude of improvement over f_best.
    Dividing by cost penalizes expensive evaluations, steering the search
    towards cheaper regions of the domain while still exploiting GP
    posterior uncertainty.
    
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
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Handle potential overflow/underflow in norm.cdf/pdf
    # z can be large. norm.cdf is stable. norm.pdf might underflow to 0 for large |z|.
    # If z is very large positive (mu << f_best), EI ~ f_best - mu.
    # If z is very large negative (mu >> f_best), EI ~ 0.
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # To avoid numerical issues when sigma is very small:
    # If sigma is tiny, z is huge.
    # We can use a stable formula or just let NumPy handle it, as pdf(z) goes to 0 fast.
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # EI should be non-negative. Due to numerical errors, it might be slightly negative.
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    score = ei / cost_safe
    
    # Handle non-finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
