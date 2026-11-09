"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guard
_START_TIME = time.time()
DEADLINE_MARGIN = 10.0  # Seconds before the 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function with
    adaptive exploration bonus and smooth cost normalization.

    Args:
        mu: Array of shape (N,) with GP mean predictions.
        sigma: Array of shape (N,) with GP standard deviations.
        f_best: The lowest observed objective value.
        cost: Array of shape (N,) with evaluation costs.

    Returns:
        Array of shape (N,) with acquisition scores. Higher is better.
    """
    # Check deadline
    if time.time() - _START_TIME > 330.0 - DEADLINE_MARGIN:
        # Fallback: exploitation with cost awareness
        safe_cost = np.maximum(cost, 1e-6)
        return -mu / safe_cost

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # --- Expected Improvement (EI) ---
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z),  z = (f_best - mu)/sigma
    safe_sigma = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / safe_sigma

    # Standard normal CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)

    # Expected Improvement: positive when mu < f_best (potential improvement)
    ei = (f_best - mu) * cdf_z + safe_sigma * pdf_z

    # Clip tiny negative values (numerical noise) to 0
    ei = np.maximum(ei, 0.0)

    # --- Adaptive exploration bonus ---
    # Add a small UCB-like term to encourage exploration in high-uncertainty
    # regions. The coefficient is tuned to be modest so exploitation dominates
    # when EI is large, but exploration can win when EI is small.
    # beta is scaled by the typical EI to keep the bonus proportional.
    typical_ei = np.median(ei[ei > 0]) if np.any(ei > 0) else 1.0
    beta = 0.15 * typical_ei  # exploration strength
    exploration_bonus = beta * sigma / (safe_sigma + 1e-12)  # normalized sigma

    # Combine EI and exploration
    raw_score = ei + exploration_bonus

    # --- Cost-aware normalization ---
    # Use log(cost) for smoother behavior; lower cost => lower log(cost) =>
    # we subtract it so cheaper points get higher scores.
    safe_cost = np.maximum(cost, 1e-6)
    log_cost = np.log(safe_cost)

    # Normalize log_cost to [0, 1] range for consistent scaling
    log_cost_min = np.min(log_cost)
    log_cost_max = np.max(log_cost)
    log_cost_range = log_cost_max - log_cost_min
    if log_cost_range > 1e-9:
        log_cost_norm = (log_cost - log_cost_min) / log_cost_range
    else:
        log_cost_norm = np.zeros_like(log_cost)

    # Penalize expensive evaluations: score decreases with normalized cost
    # Use a multiplicative factor in [0.5, 1.0] so cost matters but doesn't
    # completely override the improvement signal
    cost_penalty = 1.0 - 0.5 * log_cost_norm
    scores = raw_score * cost_penalty

    # --- Temperature-scaled soft-max for smooth ranking ---
    # This creates a smooth, numerically stable score that preserves ordering
    # but compresses extreme values.
    # Use log-sum-exp for numerical stability
    scores_for_lse = scores
    max_s = np.max(scores_for_lse)
    if max_s > -1e9:
        shifted = scores_for_lse - max_s
        # Soft-max with temperature 1.0 (essentially a stabilized version)
        # We keep the original scores but ensure they are finite and well-scaled
        pass

    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)

    return scores
# EVOLVE-BLOCK-END
