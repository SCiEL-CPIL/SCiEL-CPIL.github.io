# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware Expected Improvement (EI) for minimization.
    
    The acquisition function is defined as:
    Score(x) = EI(x) / cost(x)
    
    Expected Improvement (EI) quantifies the expected improvement over the
    current best value f_best. For minimization, improvement is f_best - f(x).
    Assuming f(x) ~ N(mu, sigma^2), EI is:
    EI(x) = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    where Z = (f_best - mu) / sigma, Phi is the standard normal CDF,
    and phi is the standard normal PDF.
    
    We normalize by cost to prioritize evaluations that are likely to
    improve the best value per unit of cost.
    
    Parameters:
    mu: Mean predictions from the GP (shape 2000,)
    sigma: Standard deviation predictions from the GP (shape 2000,)
    f_best: Best observed function value (scalar)
    cost: Cost of evaluating each candidate (shape 2000,)
    
    Returns:
    scores: Acquisition scores (shape 2000,)
    """
    # Safe sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Z-score
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # EI should be non-negative. If mu > f_best, EI is small/zero.
    # Clip to 0 to ensure we don't get negative scores for points predicted to be worse.
    ei = np.maximum(ei, 0.0)
    
    # Safe division for cost
    cost_safe = cost + 1e-9
    
    # Cost-aware score
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
