"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Budget-decaying UCB with cost normalization.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (N,)
        GP mean predictions at candidate points.
    sigma : np.ndarray, shape (N,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed function value so far.
    cost : np.ndarray, shape (N,)
        Evaluation cost for each candidate point.
    
    Returns:
    --------
    np.ndarray, shape (N,)
        Acquisition values (higher = more promising).
    """
    # Safety margin: return early if we're getting close to the 300s limit
    elapsed = time.time() - _start_time
    if elapsed > 290:
        # Return a reasonable fallback: just EI without cost penalty
        mu_arr = np.asarray(mu, dtype=np.float64)
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        sigma_safe = np.maximum(sigma_arr, 1e-9)
        improvement = f_best - mu_arr
        z = improvement / sigma_safe
        ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero
    cost = np.clip(cost, 1e-8, None)
    
    # Compute z-score for Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Expected Improvement: E[improvement | improvement > 0]
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = improvement * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # UCB: mu + beta * sigma (for minimization, we want low mu and high sigma)
    # For minimization, UCB-style exploration is: -mu + beta * sigma
    # But we want higher score = better, so:
    # UCB score = -(mu) + beta * sigma  (we want to minimize mu, explore where sigma is high)
    
    # Budget-decaying beta: starts high for exploration, decays for exploitation
    # Use f_best as a proxy for progress: when f_best is large, we're early; when ~0, we're late
    # beta = beta_max * (1 - progress) + beta_min
    # progress = 1 - f_best / f_initial, but we don't know f_initial
    # Instead, use: beta = beta_max * exp(-decay * f_best)
    # When f_best is large (early), beta is small... that's wrong.
    # Let's use: beta = beta_min + (beta_max - beta_min) * (1 - min(f_best, 1.0))
    # When f_best > 1 (early), beta ~ beta_min (low exploration)
    # When f_best ~ 0 (late), beta ~ beta_max (high exploration)? That's also wrong.
    
    # Actually, for minimization, we want to explore early when we have no good info.
    # f_best starts high (bad) and decreases toward 0 (good).
    # So early = f_best high, late = f_best low.
    # We want high beta early, low beta late.
    # beta = beta_min + (beta_max - beta_min) * max(0, 1 - f_best)
    # When f_best >= 1: beta = beta_min (low exploration, but f_best >= 1 means we're early)
    # That's inverted. Let's think again.
    
    # Early: f_best is large (e.g., 50), we should explore more -> high beta
    # Late: f_best is small (e.g., 0.1), we should exploit more -> low beta
    # beta = beta_min + (beta_max - beta_min) * (1 / (1 + f_best))
    # When f_best is large: beta ~ beta_min
    # When f_best is small: beta ~ beta_max
    # Still inverted!
    
    # Let's just use a fixed moderate beta and rely on EI/cost for the main decision.
    # Or use: beta = 2.0 * (1 + 1/f_best) when f_best > 0.1, else 1.0
    # Actually, let's use a simpler approach:
    # beta = 1.5 + 0.5 * max(0, 1 - f_best)
    # This gives beta in [1.5, 2.0], with higher exploration when f_best is close to 0.
    # Wait, that's still backwards. When f_best is close to 0, we've found a good point,
    # so we should exploit (lower beta). When f_best is large, we should explore (higher beta).
    
    # Correct: beta = beta_min + (beta_max - beta_min) * min(1, f_best / 10)
    # When f_best >= 10: beta = beta_max (high exploration)
    # When f_best = 0: beta = beta_min (low exploration, exploit)
    beta_min = 0.5
    beta_max = 2.5
    beta = beta_min + (beta_max - beta_min) * min(1.0, max(0.0, f_best / 10.0))
    
    # UCB score for minimization: -mu + beta * sigma
    # Higher is better (we want low mu, high sigma)
    ucb_score = -mu + beta * sigma_safe
    
    # Combine EI and UCB
    # EI is in objective units (improvement), UCB is a mix of mu and sigma
    # Normalize both to comparable scales
    # EI max is roughly f_best, UCB range depends on mu and sigma
    
    # Cost-aware: divide by cost
    # Use a sublinear cost penalty to avoid over-penalizing higher-cost regions
    cost_penalty = np.sqrt(cost)
    cost_penalty = np.maximum(cost_penalty, 1e-8)
    
    # Score: (alpha * EI + (1-alpha) * UCB) / cost_penalty
    # alpha decays as we find better solutions: early = more UCB (exploration), late = more EI (exploitation)
    alpha_ei = 0.5 + 0.5 * min(1.0, max(0.0, 1.0 - f_best / 5.0))
    alpha_ucb = 1.0 - alpha_ei
    
    combined = alpha_ei * ei + alpha_ucb * ucb_score
    
    # Ensure combined is finite
    combined = np.where(np.isfinite(combined), combined, 0.0)
    
    # Divide by cost penalty
    score = combined / cost_penalty
    
    # Add a small exploration bonus for low-cost regions early on
    # This helps find the global minimum which may be in a higher-cost region
    # The bonus decays as f_best improves
    exploration_scale = max(0.0, min(f_best, 2.0)) * 0.02
    cost_range = cost - cost.min()
    cost_range_max = max(cost_range.max(), 1e-8)
    low_cost_bonus = (1.0 - cost_range / cost_range_max) * exploration_scale
    score = score + low_cost_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
