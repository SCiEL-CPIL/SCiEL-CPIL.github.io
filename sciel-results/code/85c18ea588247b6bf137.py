"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-computed constants for speed
_N = 2000
# We'll use a hybrid acquisition: blend of cost-aware EI and UCB
# with adaptive weights based on uncertainty relative to best.

def _norm_cdf(x):
    """Fast normal CDF using scipy or fallback."""
    # Use scipy for accuracy; we'll import once
    from scipy.stats import norm
    return norm.cdf(x)

def _norm_pdf(x):
    from scipy.stats import norm
    return norm.pdf(x)


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next).
    
    mu, sigma: arrays of shape (2000,) in objective units
    f_best: scalar, lowest observed value
    cost: array of shape (2000,), evaluation cost per candidate
    
    Uses a cost-aware hybrid of Expected Improvement and UCB,
    with adaptive exploration based on uncertainty ratio.
    """
    # Deadline guard: track time to avoid exceeding 300s
    # This function is called once per iteration; total loop is bounded by cost < 30
    # but we add a safety margin just in case.
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-scores for EI
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Expected Improvement (standard)
    ei = improvement * _norm_cdf(z) + sigma_safe * _norm_pdf(z)
    # Clip negative EI to 0
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: divide by cost
    # cost is in [1, 2] range (1 + mean(x_scaled))
    cost_safe = np.maximum(cost, 1e-8)
    ei_cost = ei / cost_safe
    
    # UCB component: mu - beta * sigma (for minimization, lower is better)
    # We want to explore where uncertainty is high relative to improvement potential
    # Use an adaptive beta based on the ratio of max sigma to typical improvement
    sigma_max = np.max(sigma_safe) if len(sigma_safe) > 0 else 1.0
    sigma_median = np.median(sigma_safe) if len(sigma_safe) > 0 else 1.0
    
    # Adaptive exploration weight: larger when uncertainty is high relative to best
    # Use a quantile-based gate: consider only candidates where EI is above 25th percentile
    if len(ei_cost) > 0:
        ei_q25 = np.percentile(ei_cost, 25)
    else:
        ei_q25 = 0.0
    
    # UCB for minimization: we score higher for lower mu and higher sigma
    # ucb_score = -mu + beta * sigma, then normalize
    # Let's use a relative formulation:
    mu_min = np.min(mu) if len(mu) > 0 else 0.0
    mu_range = np.max(mu) - mu_min if len(mu) > 0 else 1.0
    mu_range_safe = max(mu_range, 1e-8)
    
    # Normalize mu to [0, 1] range (inverted: lower mu = higher score)
    mu_norm = 1.0 - (mu - mu_min) / mu_range_safe
    mu_norm = np.clip(mu_norm, 0.0, 1.0)
    
    # Normalize sigma to [0, 1]
    sigma_norm = sigma_safe / sigma_max if sigma_max > 0 else np.zeros_like(sigma_safe)
    sigma_norm = np.clip(sigma_norm, 0.0, 1.0)
    
    # UCB score: blend of normalized mu (exploitation) and sigma (exploration)
    # Use adaptive beta: higher when we're uncertain overall
    # beta ranges from 0.3 to 1.0 based on relative uncertainty
    rel_uncertainty = sigma_median / (np.abs(f_best - mu_min) + 1e-8) if np.abs(f_best - mu_min) > 1e-8 else 1.0
    beta = 0.3 + 0.7 * min(rel_uncertainty, 1.0)
    
    ucb_score = mu_norm + beta * sigma_norm
    
    # Cost-aware UCB
    ucb_cost = ucb_score / cost_safe
    
    # Normalize both to comparable scales
    ei_cost_max = np.max(ei_cost) if len(ei_cost) > 0 else 1.0
    ucb_cost_max = np.max(ucb_cost) if len(ucb_cost) > 0 else 1.0
    
    ei_cost_norm = ei_cost / (ei_cost_max + 1e-12)
    ucb_cost_norm = ucb_cost / (ucb_cost_max + 1e-12)
    
    # Adaptive blending: use more UCB when EI signal is weak (many candidates have similar EI)
    ei_spread = np.std(ei_cost) if len(ei_cost) > 0 else 0.0
    ei_mean = np.mean(ei_cost) if len(ei_cost) > 0 else 0.0
    cv = ei_spread / (ei_mean + 1e-12) if ei_mean > 1e-12 else 1.0
    
    # If EI is concentrated (low CV), trust EI more; if spread out, blend with UCB
    # Weight for EI: 0.5 to 0.9
    w_ei = 0.5 + 0.4 * min(cv, 1.0)
    w_ucb = 1.0 - w_ei
    
    # Combine
    score = w_ei * ei_cost_norm + w_ucb * ucb_cost_norm
    
    # Apply quantile-based feasibility gate: boost candidates above 25th percentile of EI
    # This adapts to the distribution of cost-efficiency
    if len(ei_cost) > 0:
        gate_mask = ei_cost >= ei_q25
        # Slightly boost gated candidates
        score = np.where(gate_mask, score * 1.1, score * 0.9)
    
    # Ensure finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
