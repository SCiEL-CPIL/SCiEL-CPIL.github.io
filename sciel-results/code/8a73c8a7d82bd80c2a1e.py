"""Cost-aware acquisition for Griewank 2D BO with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard: 330s wall-clock limit, use 315s as safety margin.
_DEADLINE = time.time() + 315.0
_EPS = 1e-9


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite acquisition scores (higher = evaluate next).
    
    Uses a blended strategy:
      - Cost-normalized Expected Improvement (EI)
      - Cost-normalized Upper Confidence Bound (UCB)
      - Pure exploitation (negative mean) weighted by confidence
    
    The blend is tuned to balance exploration (uncertainty) and exploitation
    (low predicted mean) while respecting the per-point evaluation cost.
    """
    # Ensure we return before the hard deadline.
    if time.time() > _DEADLINE - 2.0:
        # Fallback: return pure exploitation (negative mean) as a safe, fast answer.
        return -mu.copy()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Guard against zero/negative costs and zero/negative sigma.
    cost_safe = np.maximum(cost, _EPS)
    sigma_safe = np.maximum(sigma, _EPS)

    # --- Expected Improvement (EI) ---
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu)/sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Clip negative EI to zero (EI should be non-negative when f_best >= mu, but numerically safe)
    ei = np.maximum(ei, 0.0)
    ei_cost = ei / cost_safe

    # --- UCB (Upper Confidence Bound for minimization) ---
    # We want to explore where mu is low (good) and sigma is high (uncertain).
    # UCB_score = -mu + beta * sigma  (higher is better for selection)
    # Normalize by cost.
    beta = 1.0
    ucb = -mu + beta * sigma_safe
    ucb_cost = ucb / cost_safe

    # --- Pure exploitation ---
    # Simply prefer low predicted mean, normalized by cost.
    exploit = -mu / cost_safe

    # --- Blend ---
    # The knowledge suggests blending EI, UCB, and exploitation with cost-normalization
    # improved score from 1.0 to 1.52. We use a weighted sum.
    # Weights: EI is the primary signal, UCB adds exploration, exploitation anchors.
    w_ei = 0.5
    w_ucb = 0.3
    w_exploit = 0.2

    score = w_ei * ei_cost + w_ucb * ucb_cost + w_exploit * exploit

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
