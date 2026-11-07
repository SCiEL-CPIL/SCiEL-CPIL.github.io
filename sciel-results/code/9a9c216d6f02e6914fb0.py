"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track time to ensure we return well before the limit.
_DEADLINE = 320.0  # safety margin of 10s
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Powell 4D.
    
    Structural change: Instead of simple CAEI, we use a hybrid approach:
    1. Compute Expected Improvement (EI) for minimization.
    2. Apply a cost penalty that is adaptive based on the current f_best.
       - When f_best is high (early), we are more willing to pay higher costs for exploration.
       - When f_best is low (late), we heavily penalize cost to exploit cheap, high-confidence points.
    3. Add an uncertainty bonus that is also cost-adjusted.
    
    The key insight from the knowledge base is that a dynamic log-cost penalty
    that decays as f_best improves allows early exploration of expensive regions
    and late-stage exploitation of cheap high-confidence points.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure we haven't exceeded the deadline
    if _elapsed() > _DEADLINE:
        # Return a safe fallback: just use 1/cost to at least prefer cheap points
        cost_safe = np.maximum(cost, 1e-6)
        return np.where(np.isfinite(1.0 / cost_safe), 1.0 / cost_safe, 0.0)
    
    # Ensure cost is at least a small positive number
    cost_safe = np.maximum(cost, 1e-6)
    
    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clamp EI to non-negative
    ei = np.maximum(ei, 0.0)
    
    # Dynamic cost penalty:
    # The penalty should be stronger when f_best is low (we're close to the optimum)
    # and weaker when f_best is high (we're still exploring).
    # 
    # Use a log-based penalty that adapts to f_best:
    # penalty = log(1 + cost) * (1 + |f_best|)
    # 
    # When f_best is large (early), the multiplier is large, but we want LESS penalty,
    # so we actually want the penalty to DECREASE as f_best improves (gets smaller).
    # 
    # Let's think differently:
    # - Early: f_best ~ 10-100, we want to allow higher costs. Penalty should be mild.
    # - Late: f_best ~ 0.001-0.1, we want to avoid high costs. Penalty should be strong.
    #
    # So the penalty strength should INCREASE as f_best DECREASES.
    # penalty_strength = 1 / (1 + f_best)  -- this is large when f_best is small
    #
    # Actually, let's use: cost_penalty = cost_safe ** (1 / (1 + max(f_best, 0.01)))
    # When f_best is large, exponent ~ 1, so penalty ~ cost
    # When f_best is small, exponent ~ 1, so penalty ~ cost
    # This doesn't change much. Let me try a different approach.
    
    # Use a multiplicative cost factor that depends on f_best:
    # cost_factor = 1 + alpha * cost_safe * (1 / (1 + f_best * scale))
    # where scale normalizes f_best to be in a reasonable range.
    
    # Powell 4D minimum is 0, and initial values can be ~10-100.
    # Let's use: effective_cost = cost_safe * (1 + beta / (1 + f_best))
    # When f_best is large, beta/(1+f_best) ~ 0, so effective_cost ~ cost
    # When f_best is small, beta/(1+f_best) ~ beta, so effective_cost ~ cost * (1+beta)
    # This makes late-stage costs more expensive, encouraging cheap evaluations.
    
    # Choose beta based on the expected range of f_best
    beta = 2.0
    
    # Compute effective cost
    f_best_safe = max(f_best, 0.001)
    effective_cost = cost_safe * (1.0 + beta / (1.0 + f_best_safe))
    
    # Cost-Adjusted EI
    caei = ei / effective_cost
    
    # Uncertainty bonus, also cost-adjusted
    alpha = 0.1
    exploration_term = alpha * sigma_safe / effective_cost
    
    score = caei + exploration_term
    
    # Ensure finite outputs
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline again before returning
    if _elapsed() > _DEADLINE:
        return np.where(np.isfinite(1.0 / cost_safe), 1.0 / cost_safe, 0.0)
    
    return score
# EVOLVE-BLOCK-END
