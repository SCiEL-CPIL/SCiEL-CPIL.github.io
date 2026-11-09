import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT = 285


def _elapsed_ok():
    return (time.time() - _START_TIME) < _TIMEOUT


def acquisition(mu, sigma, f_best, cost):
    # Sanitize inputs
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    cost = np.maximum(cost, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)

    # ---- Lagrangian cost-aware score ----
    # Core idea: score = (f_best - mu) - lambda * cost
    # This is a constrained optimization relaxation where lambda tunes
    # the trade-off between improvement and cost.
    # 
    # We dynamically tune lambda so that the average cost of the "winning"
    # candidates (those with positive expected improvement) stays near a target.
    # This decouples exploration geometry from explicit cost division.

    # Expected improvement (positive when mu < f_best)
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Identify candidates with meaningful expected improvement
    # Use a threshold relative to the EI scale
    ei_scale = np.percentile(ei, 90) if np.max(ei) > 1e-12 else 1.0
    threshold = 0.1 * ei_scale
    mask = ei > threshold

    if np.any(mask):
        # Estimate the "effective" cost of promising candidates
        mean_cost_promising = np.mean(cost[mask])
        # Target: we want the average cost of selected candidates to be
        # around 1.5 (mid-range of [1,2]), so lambda penalizes expensive points
        # when the promising set is already expensive.
        # lambda = alpha * (mean_cost - target_cost) / ei_scale
        # If mean_cost > target, increase lambda (more penalty on cost)
        # If mean_cost < target, decrease lambda (less penalty)
        target_cost = 1.3
        alpha = 1.0  # scaling factor
        lambda_val = alpha * (mean_cost_promising - target_cost) / (ei_scale + 1e-9)
        # Clamp lambda to avoid extreme values
        lambda_val = np.clip(lambda_val, -0.5, 2.0)
    else:
        # No clearly promising candidates; use a moderate default
        lambda_val = 0.5

    # Base score: expected improvement minus cost penalty
    # This is the Lagrangian relaxation: maximize EI - lambda * cost
    score = ei - lambda_val * cost

    # ---- Adaptive UCB component for exploration ----
    # Add a UCB term to encourage exploration in uncertain regions
    # This prevents the algorithm from getting stuck in local optima
    # when EI is zero everywhere (e.g., early on or in flat regions)
    
    sigma_max = np.max(sigma)
    sigma_med = np.median(sigma) + 1e-9
    beta = 1.0 + (sigma_max / sigma_med)
    beta = np.clip(beta, 1.0, 5.0)
    
    # UCB score for minimization: -mu + beta * sigma
    ucb_score = -mu + beta * sigma
    
    # Normalize UCB to be comparable with EI scale
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-12:
        ucb_norm = (ucb_score - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(mu)
    
    # Blend EI-based score with UCB exploration
    # When EI is low (no clear improvement), lean more on UCB
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
        # Weight: higher when EI is meaningful
        w_ei = 0.7 + 0.3 * np.clip(ei_max / (ei_scale + 1e-9), 0, 1)
    else:
        w_ei = 0.3
    
    w_ucb = 1.0 - w_ei
    
    # Combine
    score = w_ei * (score / (ei_scale + 1e-9)) + w_ucb * ucb_norm
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Fallback for near-timeout
    if not _elapsed_ok():
        fallback = ei / np.power(cost, 0.7)
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        return fallback
    
    return score
