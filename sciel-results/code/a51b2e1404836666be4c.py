"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track wall-clock deadline (330s limit, use 310s margin)
_T0 = time.time()
_DEADLINE = 310.0


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization of Styblinski-Tang 2D.
    
    mu, sigma, cost: arrays of shape (2000,)
    f_best: scalar, lowest observed value
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard
    if not _time_ok():
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / np.maximum(cost, 1e-6)
        return np.where(np.isfinite(score), score, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: ensure finite and positive
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    sigma_safe = np.maximum(sigma, 1e-8)

    # Infer progress based on f_best
    f_best_val = float(f_best)
    if not np.isfinite(f_best_val):
        f_best_val = 0.0
    # ST-2d: f(0,0)=0, min=-78.3323
    # progress goes from 0 to 1 as f_best goes from 0 to -78.3323
    progress = np.clip(-f_best_val / 78.3323, 0.0, 1.0)

    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Compute Probability of Improvement (POI) - more robust than EI for small budgets
    poi = norm.cdf(z)

    # Compute UCB
    # Adaptive beta: higher early for exploration, lower late for exploitation
    beta = 4.0 * (1.0 - progress) + 0.5
    ucb = -mu + beta * sigma_safe

    # Cost-aware scoring
    cost_safe = np.maximum(cost, 1e-6)

    # Normalize cost to [0,1] range for better scaling
    # Cost ranges from 1 (center) to 2 (corners)
    cost_normalized = (cost_safe - 1.0) / 1.0  # 0 to 1

    # Use a cost discount that's more aggressive than simple division
    # This strongly prefers cheaper evaluations
    cost_discount = 1.0 / (1.0 + 2.0 * cost_normalized)  # 1.0 at center, 0.33 at corners

    # Blend EI and UCB based on progress
    # Early: favor UCB for exploration to find shifted minima
    # Late: favor EI for exploitation to refine the minimum
    w_ucb = 0.6 * (1.0 - progress) + 0.2
    w_ei = 1.0 - w_ucb

    # Scale EI by a factor to make it comparable to UCB
    # The scale of EI depends on the problem; normalize by typical improvement
    ei_scale = np.maximum(np.percentile(ei, 95), 1e-6)
    ei_normalized = ei / ei_scale

    ucb_scale = np.max(np.abs(ucb)) if np.max(np.abs(ucb)) > 0 else 1.0
    ucb_normalized = ucb / ucb_scale

    # Combine with cost discount
    score_ei = w_ei * ei_normalized * cost_discount
    score_ucb = w_ucb * ucb_normalized * cost_discount

    score = score_ei + score_ucb

    # Add a small exploration bonus for high-uncertainty regions
    # This helps escape local optima and explore the domain
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    uncertainty_bonus = 0.15 * (sigma_safe / sigma_max) * (1.0 - progress) * cost_discount
    score = score + uncertainty_bonus

    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
