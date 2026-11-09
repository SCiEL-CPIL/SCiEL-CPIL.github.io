import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Effective Global Optimization (CEGO).
    
    The CEGO criterion maximizes the expected improvement divided by the expected cost.
    For a Gaussian process with mean mu and standard deviation sigma, and a constant 
    cost c for a point, the optimal acquisition value is:
    
    Q(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    
    Since we must return scores that the evaluator will use to pick the best candidate,
    and the evaluator picks the candidate with the highest score, we return Q(x) / cost(x).
    This directly optimizes the ratio of expected improvement to cost.
    
    Args:
        mu: array of GP mean predictions, shape (2000,)
        sigma: array of GP std predictions, shape (2000,)
        f_best: current best observed value
        cost: array of evaluation costs, shape (2000,)
    
    Returns:
        array of acquisition scores, shape (2000,), higher = prefer
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.nan_to_num(sigma, nan=1e-12, posinf=1.0, neginf=1e-12)
    mu = np.nan_to_num(mu, nan=f_best, posinf=1e6, neginf=-1e6)
    cost = np.maximum(cost, 1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    improvement = f_best - mu
    z = improvement / sigma
    
    # Handle numerical stability for large z
    # If z is very large positive, Phi(z) ~ 1, phi(z) ~ 0, EI ~ f_best - mu
    # If z is very large negative, Phi(z) ~ 0, phi(z) ~ 0, EI ~ 0
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (it should be by definition, but floating point errors can occur)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # This is the CEGO criterion: maximize expected improvement per unit cost
    scores = ei / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
