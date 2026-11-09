"""Rewritten program using a cost-aware Expected Improvement (EI) acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function.
    
    For minimization, Expected Improvement measures the expected amount by which
    the function value at x is lower than the current best f_best.
    
    EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((mu - f_best)/sigma)
    
    We divide by cost to get cost-aware EI (cEI).
    Higher scores indicate better candidates to evaluate.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate the standardized improvement u
    # u = (f_best - mu) / sigma
    # If mu < f_best, u > 0 (improvement expected)
    # If mu > f_best, u < 0 (no improvement expected)
    u = (f_best - mu) / sigma_safe
    
    # Calculate PDF and CDF of the standard normal distribution at u
    # norm.cdf(u) = Phi(u)
    # norm.pdf(u) = phi(u)
    # Note: phi(u) = phi(-u), so we can use norm.pdf(u) directly
    # The formula for EI is:
    # EI = sigma * (u * Phi(u) + phi(u))
    # This is derived from E[max(0, f_best - Y)] where Y ~ N(mu, sigma^2)
    
    cdf_vals = norm.cdf(u)
    pdf_vals = norm.pdf(u)
    
    # Calculate Expected Improvement
    ei = sigma_safe * (u * cdf_vals + pdf_vals)
    
    # Numerical stability: EI should be non-negative
    # For very negative u, the term (u * Phi(u) + phi(u)) can suffer from cancellation
    # or become slightly negative due to floating point errors.
    # We clamp to 0.
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI per unit cost
    # cost is given, and is >= 1.0
    scores = ei / cost
    
    # Ensure all outputs are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
