"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement: P(f(x) < f_best)
    # Since we are minimizing, f(x) < f_best is an improvement.
    # The GP predicts mu and sigma. f(x) ~ N(mu, sigma^2).
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)
    
    # Mean improvement (expected value of f_best - f(x) if it's positive, but here we just use the raw difference weighted by prob)
    # The directive specifies: P(mu < f_best) * (f_best - mu)
    # Note: if mu > f_best, (f_best - mu) is negative, and pi is small. The product will be negative.
    # This effectively penalizes points predicted to be worse than f_best.
    weighted_improvement = (f_best - mu) * pi
    
    # Cost-aware score
    # Ensure cost is not zero
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = weighted_improvement / cost_safe
    
    # Return finite scores. If any NaN or Inf, replace with -inf or 0?
    # "Return 2000 finite scores ... anything else invalidates the whole result"
    # So we must ensure finiteness.
    # If mu or sigma or cost are bad, we might get issues.
    # Assuming inputs are valid from the evaluator.
    # But to be safe:
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
