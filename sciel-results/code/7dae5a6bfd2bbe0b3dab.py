import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value so far
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: use a softer cost penalty
    # cost is in [1, 2], so log(1 + cost) gives a gentler penalty
    # Alternatively, use cost^0.5 for a moderate penalty
    cost_penalty = np.sqrt(cost)
    
    # Base acquisition: EI / cost_penalty
    acq = ei / cost_penalty
    
    # Add small exploration bonus for high-uncertainty regions
    # This helps escape local minima in Styblinski-Tang
    exploration_weight = 0.1
    acq += exploration_weight * sigma_safe / cost_penalty
    
    # Ensure all values are finite
    acq = np.where(np.isfinite(acq), acq, 0.0)
    
    return acq
