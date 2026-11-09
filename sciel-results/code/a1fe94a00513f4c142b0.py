# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware Expected Improvement (EI) for minimization.
    
    The acquisition function is defined as:
    Score(x) = EI(x) / cost(x)
    
    EI(x) represents the expected improvement over the current best f_best.
    For minimization:
    If sigma > 0:
        z = (f_best - mu) / sigma
        EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    If sigma == 0:
        EI = max(f_best - mu, 0)
    
    Parameters:
    mu: Mean predictions from the GP (shape 2000,)
    sigma: Standard deviation predictions from the GP (shape 2000,)
    f_best: Best observed function value (scalar)
    cost: Cost of evaluating each candidate (shape 2000,)
    
    Returns:
    scores: Acquisition scores (shape 2000,)
    """
    # Calculate z-score
    # Avoid division by zero by adding a small epsilon to sigma
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    z = (f_best - mu) / sigma_safe
    
    # Calculate PDF and CDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # We use sigma_safe in the first term for numerical stability, but mathematically it's (f_best - mu)
    # Note: As sigma -> 0, (f_best - mu) * Phi(z) + sigma * phi(z) approaches max(f_best - mu, 0)
    # However, using sigma_safe in the product can cause issues if mu is far from f_best and sigma is tiny.
    # Standard formula: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Let's stick to the standard formula but handle the sigma=0 case explicitly if needed, 
    # or rely on the limit which is stable if we are careful.
    
    # A more numerically stable way often used:
    # If z is very negative, Phi(z) ~ 0, phi(z) ~ 0 -> EI ~ 0
    # If z is very positive, Phi(z) ~ 1, phi(z) ~ 0 -> EI ~ f_best - mu
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical errors might make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware normalization
    cost_safe = cost + 1e-9
    
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
