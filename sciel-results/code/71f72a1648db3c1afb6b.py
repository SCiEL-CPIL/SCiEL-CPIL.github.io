"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard: each acquisition call must finish well under 300 s.
# We track a per-call budget conservatively.
_CALL_DEADLINE = 280.0  # seconds, safety margin under 300 s


def _norm_cdf(x):
    return norm.cdf(x)


def _norm_pdf(x):
    return norm.pdf(x)


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite acquisition scores (higher = evaluate next).

    Strategy: hybrid of cost-normalized Expected Improvement (EI) and a
    Thompson-sampling-like exploration term (UCB), with progress-based
    weighting. Early iterations favour exploration (UCB) to find multiple
    basins in Shekel 4d's multimodal landscape; later iterations shift
    towards exploitation (EI) to refine the best basin. Cost normalization
    ensures we prefer cheaper evaluations, conserving budget.

    mu, sigma, f_best, cost: arrays of shape (2000,).
    """
    t0 = time.time()
    deadline = t0 + _CALL_DEADLINE

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure finite, positive cost (avoid division by zero)
    cost = np.where(cost > 1e-12, cost, 1.0)

    # Clamp sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)

    # Progress indicator: how much of the budget we've used.
    # cost is per-candidate; total spent is tracked externally, but we can
    # infer progress from f_best relative to the known minimum (-10.5364).
    # A simpler proxy: use the number of evaluations implied by cost sum,
    # but we don't have that. Instead, use a heuristic based on sigma scale.
    #
    # Better: use the spread of mu values as a proxy for exploration progress.
    # Early on, mu is noisy and spread out; later it's concentrated.
    mu_range = np.ptp(mu)
    # Normalize: if mu_range is large, we're still exploring.
    # Shekel 4d range is roughly [-10.5, 10.5], so range ~21.
    progress = 1.0 - np.clip(mu_range / 20.0, 0.0, 1.0)
    # progress in [0, 1]: 0 = early (explore), 1 = late (exploit)

    # Weight for exploration vs exploitation
    # Early: weight_ucb high, weight_ei low
    # Late: weight_ucb low, weight_ei high
    w_ucb = 1.0 - progress
    w_ei = progress

    # Ensure at least some exploration always
    w_ucb = max(w_ucb, 0.15)
    w_ei = max(w_ei, 0.15)
    total_w = w_ucb + w_ei
    w_ucb /= total_w
    w_ei /= total_w

    # --- Expected Improvement (EI) ---
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to avoid extreme values causing numerical issues
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * _norm_cdf(z) + sigma * _norm_pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-normalized EI
    ei_cost = ei / cost

    # --- UCB exploration term ---
    # UCB = mu + beta * sigma  (but we want lower is better, so we flip)
    # For minimization, UCB score = -mu + beta * sigma (higher score = more promising)
    # Or equivalently: we want to pick points with low mu and high sigma.
    # Score_ucb = -mu + beta * sigma
    # But we need to scale this to be comparable to ei_cost.
    
    # Normalize sigma to [0, 1] for the beta term
    sigma_max = np.max(sigma)
    if sigma_max > 1e-12:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma)

    # Beta: exploration strength, decreases with progress
    beta = 2.0 * w_ucb + 0.5 * w_ei  # between 0.5 and 2.0

    # UCB score (higher = more promising for minimization)
    ucb_score = -mu + beta * sigma

    # Normalize ucb_score to be comparable to ei_cost
    # Use robust scaling: median and IQR
    ucb_med = np.median(ucb_score)
    ucb_iqr = np.percentile(ucb_score, 75) - np.percentile(ucb_score, 25)
    if ucb_iqr > 1e-12:
        ucb_norm = (ucb_score - ucb_med) / (ucb_iqr + 1e-12)
    else:
        ucb_norm = np.zeros_like(ucb_score)

    # Normalize ei_cost similarly
    ei_med = np.median(ei_cost)
    ei_iqr = np.percentile(ei_cost, 75) - np.percentile(ei_cost, 25)
    if ei_iqr > 1e-12:
        ei_norm = (ei_cost - ei_med) / (ei_iqr + 1e-12)
    else:
        ei_norm = np.zeros_like(ei_cost)

    # Combine
    score = w_ei * ei_norm + w_ucb * ucb_norm

    # Check deadline
    if time.time() > deadline - 5.0:
        # Return what we have, ensure finite
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
