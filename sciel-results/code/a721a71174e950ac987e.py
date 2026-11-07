"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware LCB acquisition function for Bayesian optimization.
    Uses an adaptive exploration coefficient (beta) that decreases as f_best improves,
    normalized by evaluation cost.
    
    Score = (beta * sigma - mu) / cost
    """
    start_time = time.time()
    DEADLINE = 320.0  # Safety margin before 330s kill

    n = mu.shape[0]
    
    # Clamp sigma for numerical stability
    sigma_safe = np.clip(sigma, 1e-12, None)
    
    # Clamp cost for numerical stability
    cost_safe = np.maximum(cost, 1e-10)
    
    # Adaptive beta with logarithmic schedule:
    # - Uses log(1 + f_best) for smoother decay than sqrt
    # - Multiplier of 2.5 for stronger early exploration
    # - Floor at 1.0 ensures exploitation when f_best is near 0
    f_best_clamped = max(f_best, 0.0)
    beta = 1.0 + 2.5 * np.log1p(f_best_clamped)
    
    # Lower Confidence Bound value: mu - beta * sigma
    # We want to minimize this value.
    # Acquisition score (higher is better) is the negative of LCB divided by cost.
    # score = - (mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    
    # Use a stronger cost penalty: cost^1.5 to strongly favor cheap evaluations
    # This is important because total budget is limited (30) and each eval costs 1+mean(x_scaled)
    cost_weighted = np.power(cost_safe, 1.5)
    
    # LCB-based score
    lcb_score = (beta * sigma_safe - mu) / cost_weighted
    
    # Add a small expected-improvement bonus to break ties in high-uncertainty regions
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # This helps the optimizer focus on points likely to improve f_best
    z = (f_best - mu) / sigma_safe
    # Use scipy's normal CDF and PDF for accuracy
    from scipy.stats import norm
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    ei_bonus = (f_best - mu) * Phi + sigma_safe * phi
    # Scale EI bonus to be small relative to LCB score
    scores = lcb_score + 0.1 * ei_bonus / cost_weighted
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Fallback: simple cost-aware LCB
        # This is extremely fast, so this branch is unlikely to be triggered
        lcb = mu - 2.0 * sigma_safe
        simple = -lcb / np.power(cost_safe, 1.5)
        return np.where(np.isfinite(simple), simple, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
