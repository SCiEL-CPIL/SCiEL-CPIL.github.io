"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # 5s safety margin before 330s kill


def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Adaptive cost-aware acquisition: blends expected improvement (exploitation)
    with uncertainty (exploration) using an adaptive beta that grows with
    remaining budget pressure, then normalizes by cost.
    """
    # If we're close to deadline, just return a safe fallback
    if _check_deadline():
        return -mu / (cost + 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-8, None)

    # Adaptive beta: start moderate, increase as we use more budget
    # cost is in [1, 2] (1 + mean of coords in [0,1])
    # Total budget is 30, initial 5 points cost ~1.5 each => ~7.5 used
    # So remaining budget fraction gives us a signal
    # Use the mean cost as a proxy for how many evals we can still do
    mean_cost = np.mean(cost)
    # Rough estimate of remaining evals: (30 - 7.5) / mean_cost ≈ 14
    # Early on (low total cost used), explore more; late, exploit more
    # We don't know total cost directly, but f_best magnitude hints at progress
    # Better: use the spread of mu relative to f_best as a progress indicator
    # If mu is close to f_best everywhere, we're converging -> exploit
    # If mu is far from f_best, we're still exploring -> explore more

    # Expected Improvement (for minimization): EI = (f_best - mu) * Phi((f_best-mu)/sigma) + sigma * phi((f_best-mu)/sigma)
    z = (f_best - mu) / sigma
    # Clip z for numerical stability
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Adaptive beta based on how much uncertainty remains relative to scale
    # High sigma relative to (f_best - min(mu)) means we haven't explored enough
    mu_range = f_best - np.min(mu)
    if mu_range < 1e-6:
        mu_range = 1e-6
    # Normalize sigma by the improvement scale
    sigma_norm = sigma / mu_range
    # Beta: higher when uncertainty is high relative to improvement
    # Base beta of 1.5, scaled up by relative uncertainty
    beta_base = 1.5
    # Cap the adaptive component
    beta = beta_base * (1.0 + np.clip(np.mean(sigma_norm), 0.0, 2.0))

    # LCB component (exploration)
    lcb = mu - beta * sigma

    # Blend: score = -cost_adjusted_lcb + weight * cost_adjusted_ei
    # The EI term rewards points likely to improve over f_best
    # The LCB term rewards uncertain regions
    # Weight EI more heavily as we converge (lower beta means more exploitation)
    ei_weight = 1.0 / (1.0 + beta)

    # Cost normalization: divide by cost to prefer cheap evaluations
    # Use sqrt(cost) instead of cost for softer penalty
    cost_factor = np.sqrt(cost + 1e-9)

    # Combined score (higher = better)
    scores = (ei_weight * ei - (1.0 - ei_weight) * lcb) / cost_factor

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e10)

    return scores
# EVOLVE-BLOCK-END
