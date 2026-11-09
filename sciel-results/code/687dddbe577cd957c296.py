"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: evaluator kills after 330s. We track elapsed time for safety.
_START_TIME = time.time()
DEADLINE = _START_TIME + 315.0  # 15s safety margin


def _probability_of_improvement(mu, sigma, f_best):
    """
    Probability of Improvement (PI) for minimization:
        PI = Phi((f_best - mu) / sigma)
    This is bounded in [0, 1], providing a sigmoidal exploration signal
    that naturally saturates in high-uncertainty regions.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    return pi


def _expected_improvement(mu, sigma, f_best):
    """
    Standard Expected Improvement (EI) for minimization:
        EI = (f_best - mu) * Phi(z) + sigma * phi(z),  z = (f_best - mu) / sigma
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    cdf_part = improvement * norm.cdf(z)
    pdf_part = sigma_safe * norm.pdf(z)
    
    ei = cdf_part + pdf_part
    ei = np.maximum(ei, 0.0)
    return ei


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Griewank 2D.
    
    Uses a blend of Probability of Improvement (PI) and Expected Improvement (EI),
    normalized by cost. PI provides bounded, sigmoidal exploration that avoids
    unbounded noise-chasing, while EI adds exploitation pressure. The blend
    balances exploration and exploitation in the cost-aware setting.
    """
    # Check deadline - if we're close to timeout, return a safe fallback
    if time.time() > DEADLINE:
        # Fallback: use PI / cost
        pi = _probability_of_improvement(mu, sigma, f_best)
        scores = pi / np.maximum(cost, 1e-9)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Compute PI and EI
    pi = _probability_of_improvement(mu, sigma, f_best)
    ei = _expected_improvement(mu, sigma, f_best)
    
    # Blend: weight PI more heavily for its bounded exploration properties
    # Use a weighted combination that emphasizes PI's stability
    blend = 0.7 * pi + 0.3 * ei
    
    # Normalize by cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = blend / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
