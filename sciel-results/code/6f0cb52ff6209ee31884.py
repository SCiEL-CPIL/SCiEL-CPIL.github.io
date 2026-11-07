"""
Hybrid acquisition: combines Expected Improvement (EI) with a lightweight
local-refinement heuristic that biases towards points near the current best.
Uses a simple time guard to stay within the 330 s wall-clock limit.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard (330 s wall-clock limit)
_T_START = time.time()
_TIME_BUDGET = 320.0  # leave 10 s margin

def _elapsed():
    return time.time() - _T_START

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: EI + local-refinement bias.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean at candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation at candidate points.
    f_best : float
        Best observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Time guard: if we are close to the deadline, fall back to a fast score
    if _elapsed() > _TIME_BUDGET:
        # Fast fallback: simple EI without heavy processing
        eps = 1e-9
        z = (f_best - mu) / (sigma + eps)
        scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    n = len(mu)
    eps = 1e-9

    # --- Standard EI (minimization) ---
    z = (f_best - mu) / (sigma + eps)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # --- Local-refinement bias: favour points with low predicted value ---
    # The idea: once we have a decent f_best, points with mu well below f_best
    # are likely in a promising region. Add a small bonus scaled by sigma
    # to encourage fine-tuning in already-promising areas.
    # Use a soft rank-based bonus: points in the best 10% of mu get a small
    # exploration-preserving bump so we don't collapse entirely to exploitation.
    #
    # Structural change vs. pure EI: the score is no longer purely a function
    # of the GP posterior in the standard EI form; it is modulated by the
    # predicted value rank, which changes the exploration-exploitation
    # trade-off adaptively.

    # Rank by mu (lower mu = better for minimisation)
    # Use a percentile-based bonus: points in the bottom 20% of mu get a bonus
    # proportional to their sigma (preserving some exploration)
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min + eps

    # Normalise mu to [0, 1], lower is better
    mu_norm = (mu - mu_min) / mu_range  # 0 = best, 1 = worst

    # Bonus: higher for better (lower) mu, scaled by sigma
    # Use a smooth kernel: bonus = sigma * exp(-2 * mu_norm)
    # This gives a strong bonus to the best-predicted points but decays smoothly
    bonus = sigma * np.exp(-2.0 * mu_norm)

    # Blend EI and bonus. Weight adapts: early on (large sigma relative to
    # spread) use more EI; later, use more bonus.
    # Simple heuristic: weight = 1.0 / (1.0 + np.mean(sigma) / (mu_range + eps))
    sigma_mean = np.mean(sigma)
    weight = 1.0 / (1.0 + sigma_mean / (mu_range + eps))
    # Clamp weight to [0.3, 0.7] so we always have a mix
    weight = np.clip(weight, 0.3, 0.7)

    scores = weight * ei + (1.0 - weight) * bonus

    # Ensure finite
    if not np.all(np.isfinite(scores)):
        finite_mask = np.isfinite(scores)
        if np.any(finite_mask):
            mean_val = np.mean(scores[finite_mask])
            scores = np.where(finite_mask, scores, mean_val)
        else:
            scores = np.zeros_like(mu)

    return scores
# EVOLVE-BLOCK-END
