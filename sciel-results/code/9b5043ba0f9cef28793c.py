import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 318.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition function with adaptive
    exploration and a nonlinear cost penalty.
    """
    # Explicit deadline guard: fall back to a simple fast heuristic
    if time.time() - _START_TIME > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        # Simple Thompson-style fallback
        rng = np.random.RandomState()
        sample = mu + sigma_safe * rng.randn(mu.shape[0])
        scores = -sample / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    n = mu.shape[0]
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Normalize cost to [0,1] range (cost is in [1,2])
    cost_norm = (cost_safe - 1.0) / 1.0  # in [0, 1]
    # Non-linear cost penalty: higher cost regions penalized more
    cost_penalty = cost_safe ** 1.5

    # Standard EI
    z = (f_best - mu) / sigma_safe
    # Clip z for numerical stability
    z = np.clip(z, -10, 10)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)

    # Adaptive exploration bonus based on uncertainty relative to scale
    # Larger sigma -> more exploration
    # Scale sigma by a small factor to keep it proportional
    scale = np.maximum(np.abs(mu).mean() + sigma.mean(), 1e-9)
    explore_bonus = 0.1 * sigma_safe / scale

    # Combined score: EI with exploration bonus, divided by cost penalty
    # The exploration bonus encourages visiting uncertain regions
    # The cost penalty discourages expensive evaluations
    scores = (ei + explore_bonus) / cost_penalty

    # Additionally, add a small UCB-style exploration term
    # This helps in early stages when EI might be small
    # kappa controls exploration; use a modest value
    kappa = 0.5
    ucb_term = kappa * sigma_safe / scale
    scores += ucb_term / cost_penalty

    # Ensure all outputs are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
