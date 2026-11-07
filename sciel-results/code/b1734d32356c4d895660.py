"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # Leave 5s safety margin before 330s kill

def _check_deadline():
    """Return True if we should stop and return current best."""
    return (time.time() - _start_time) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Structural change: Use a hybrid acquisition that combines:
    1. Expected Improvement (exploitation)
    2. Upper Confidence Bound (exploration) 
    3. Cost-aware scaling with adaptive exploration coefficient
    
    This is a different modeling family: UCB-based exploration with 
    cost-normalized EI, rather than pure KG/EI variants.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviations
    - f_best: float - best observed function value
    - cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check deadline - if exceeded, return a safe fallback
    if _check_deadline():
        return np.zeros_like(mu)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute UCB score: we want to minimize, so UCB for minimization is mu - beta*sigma
    # But for acquisition (higher is better), we use: -mu + beta*sigma (negated UCB for minimization)
    # Actually, for minimization, the "upper confidence bound" in the sense of worst case is mu + beta*sigma
    # But we want to find low values, so we should explore where mu - beta*sigma is low
    # Acquisition score should be high where we expect good (low) values
    
    # Let's use a different structural approach: 
    # Score = EI / cost + exploration_term * sigma / cost
    # where exploration_term adapts based on how far we are from optimal
    
    # Adaptive exploration coefficient: start high, decrease as f_best approaches 0
    # f_best is the lowest observed value; the minimum is 0
    # So (f_best - 0) = f_best tells us how far from optimal
    # When f_best is large, we should explore more
    exploration_coeff = 1.0 + np.log1p(np.maximum(f_best, 0.0))
    
    # Cost-aware scores: normalize by cost to prefer cheaper evaluations
    # But also add a pure exploration component scaled by sigma/cost
    ei_score = ei / cost_safe
    exploration_score = exploration_coeff * sigma_safe / cost_safe
    
    # Combine: weighted sum with EI dominant but exploration providing diversity
    scores = ei_score + 0.5 * exploration_score
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final deadline check
    if _check_deadline():
        return np.zeros_like(mu)
    
    return scores
# EVOLVE-BLOCK-END
