"""Cost-aware acquisition function for Bayesian optimization (Rosenbrock 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global variables for deadline tracking
_START_TIME = None
_DEADLINE_LIMIT = 320.0  # margin before the 330s kill switch

def _get_elapsed_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function blending Probability of Improvement (POI)
    and Expected Improvement (EI) with a quadratic cost penalty and an
    uncertainty bonus.

    Design family: POI/EI blend (retained), refined with:
      - quadratic cost penalty (smoother than 1/cost),
      - explicit uncertainty bonus (exploration),
      - log-sum-exp soft-max for the adaptive POI/EI weighting,
      - a tight deadline guard.

    Parameters:
    mu (np.ndarray): GP posterior means, shape (N,)
    sigma (np.ndarray): GP posterior standard deviations, shape (N,)
    f_best (float): Best observed function value so far.
    cost (np.ndarray): Evaluation costs for each candidate, shape (N,)

    Returns:
    np.ndarray: Acquisition scores, shape (N,). Higher = evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Deadline guard: return a fast exploit heuristic near the time limit.
    if _get_elapsed_time() > _DEADLINE_LIMIT:
        cost_safe = np.maximum(cost, 1e-8)
        safe_scores = -mu / cost_safe
        safe_scores = np.where(np.isfinite(safe_scores), safe_scores, 0.0)
        return np.maximum(safe_scores, 0.0)

    # Numerical safety.
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    f_best = float(f_best)

    # z-score for improvement.
    z = (f_best - mu) / sigma_safe

    # Probability of Improvement: P(f(x) < f_best) for minimization.
    poi = norm.cdf(z)

    # Expected Improvement magnitude.
    phi_z = norm.pdf(z)
    ei = (f_best - mu) * poi + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)

    # Gap between current best and predicted mean (clipped for stability).
    gap = np.maximum(f_best - mu, 1e-8)

    # Normalize EI to be comparable to POI (both in ~[0,1]).
    ei_norm = ei / (ei + gap * 0.5)
    ei_norm = np.clip(ei_norm, 0.0, 1.0)

    # Adaptive weighting via log-sum-exp soft-max (smooth surrogate for min/max).
    # When uncertainty is high relative to the gap, favor POI (exploration);
    # when low, favor EI (exploitation).
    uncertainty_ratio = sigma_safe / gap
    # Soft-max of the two "preference" logits.
    logit_poi = 2.0 * (uncertainty_ratio - 1.0)   # higher when uncertain
    logit_ei = -logit_poi                          # higher when certain
    m = np.maximum(logit_poi, logit_ei)
    # Soft-max weights (sum to 1).
    exp_poi = np.exp(logit_poi - m)
    exp_ei = np.exp(logit_ei - m)
    denom = exp_poi + exp_ei
    denom = np.maximum(denom, 1e-12)
    w_poi = exp_poi / denom
    w_ei = exp_ei / denom

    # Base (uncosted) improvement score: smooth blend of POI and EI.
    base = w_poi * poi + w_ei * ei_norm

    # Explicit uncertainty bonus (rewards exploration beyond the blend).
    # Scale the bonus by the typical gap so it is in comparable units.
    bonus_scale = 0.25
    uncertainty_bonus = bonus_scale * sigma_safe / (sigma_safe + gap * 0.5)
    base = base + uncertainty_bonus

    # Quadratic cost penalty (smoother than division by cost).
    # Normalize cost so the penalty magnitude is controlled.
    cost_norm = (cost_safe - 1.0) / 1.0  # cost in [1,2]; shift to [0,1]
    cost_norm = np.clip(cost_norm, 0.0, 1.0)
    lambda_cost = 0.5
    penalty = lambda_cost * cost_norm * cost_norm

    # Final score: higher is better.
    scores = base - penalty

    # Ensure finite and non-negative (higher = better).
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
