# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a robust, quantile-based Probability of Improvement (PI)
    normalized by cost.
    
    This heuristic prioritizes candidates where the lower 1-sigma quantile of the predictive
    distribution is likely to be better than the current best, normalized by the evaluation cost.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, the best observed value so far.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores. Higher is better.
    """
    # Ensure arrays are float
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Prevent division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Z-score for P(f(x) < f_best)
    # Standard PI: z = (f_best - mu) / sigma
    # Robust Quantile-based PI: We shift the threshold to require a margin.
    # We check P(f(x) < f_best - margin). Let margin = 0.5 * sigma (adaptive) or fixed.
    # Using a fixed shift in z-space is equivalent to checking a quantile.
    # Let's use a shift of 1.0 in z-space, which corresponds to the 16th percentile (lower 1-sigma bound).
    # z_robust = (f_best - mu) / sigma - 1.0
    # If z_robust is large, it means the 1-sigma lower bound is well below f_best.
    
    z_base = (f_best - mu) / sigma_safe
    
    # Apply a robustness shift. 
    # A shift of 0.5 to 1.0 is typical for "exploration-exploitation" balance in PI variants.
    # We use 0.5 to be less aggressive than 1.0, but more robust than 0.0.
    shift = 0.5
    z_robust = z_base - shift
    
    # Probability of Improvement (Robust)
    pi_robust = norm.cdf(z_robust)
    
    # Prevent division by zero in cost
    cost_safe = np.maximum(cost, 1e-8)
    
    # Score: Ratio of robust probability of improvement to cost
    scores = pi_robust / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
