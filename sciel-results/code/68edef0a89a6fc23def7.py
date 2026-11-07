"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time to track total elapsed time for the whole run
_START_TIME = time.time()
TIME_LIMIT = 330.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    
    Structural Change:
    Replaces the simple EI - beta*sigma*cost formulation with a
    Cost-Effective Expected Improvement (CEI) approach that directly
    optimizes improvement per unit cost, combined with a dynamic
    exploration bonus based on the current confidence level.
    
    The key insight is that we should not just penalize cost, but
    normalize the expected improvement by cost to get a "return on
    investment" metric. This is further adjusted by a confidence-based
    term that encourages exploration early and exploitation late.
    """
    # Deadline Guard: Check if we are close to the time limit.
    # If so, return a simple, fast score to allow the loop to finish gracefully.
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT - 5.0:
        # Fallback: Simple Expected Improvement, ignoring cost to save time/logic
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.maximum(ei, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Expected Improvement for minimization: E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Dynamic Confidence based on f_best
    # We map f_best to a "Confidence" score C in [0, 1].
    # High f_best -> Low Confidence -> More exploration
    # Low f_best  -> High Confidence -> More exploitation
    
    a_param = 0.5
    b_param = 5.0
    
    # Clamp f_best for numerical stability in exp
    f_best_clamped = np.clip(f_best, -100, 100)
    
    confidence = 1.0 / (1.0 + np.exp(a_param * (f_best_clamped - b_param)))
    
    # Cost-Effective Improvement: EI normalized by cost
    # This gives us "improvement per unit cost"
    cei = ei / cost_safe
    
    # Exploration bonus: encourage visiting uncertain regions
    # When confidence is low (early), we want to explore more
    # When confidence is high (late), we want to exploit more
    
    # The exploration bonus is proportional to sigma and inversely proportional to cost
    # But it should be damped as confidence increases
    exploration_weight = (1.0 - confidence) * 0.5
    exploration_bonus = exploration_weight * sigma_safe / cost_safe
    
    # Combine CEI and exploration bonus
    # The CEI term is the primary driver, while exploration helps early on
    score = cei + exploration_bonus
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
