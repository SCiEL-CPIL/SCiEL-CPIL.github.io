"""Cost-aware acquisition for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = _START_TIME + 295  # safety margin before 300s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining LCB with Expected Improvement (EI).
    
    Uses a hybrid score that balances:
    1. Expected Improvement (EI) - handles multimodality better than pure LCB
    2. Lower Confidence Bound (LCB) - aggressive exploration of high-variance regions
    3. Cost normalization - prefer cheaper evaluations
    
    The LCB component uses a dynamically scaled beta to encourage exploration
    in high-uncertainty, low-cost regions of the multimodal Rastrigin landscape.
    
    Args:
        mu: (2000,) GP mean predictions in objective units
        sigma: (2000,) GP std predictions in objective units
        f_best: float, best observed (lowest) value
        cost: (2000,) evaluation cost per candidate
    
    Returns:
        scores: (2000,) finite scores, higher = pick next
    """
    if time.time() > _DEADLINE:
        # Fallback: prefer cheap points with some uncertainty
        return 1.0 / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    ei = (f_best - mu) * Phi + sigma_safe * phi
    ei = np.maximum(ei, 0.0)
    
    # Compute LCB with dynamically scaled beta
    # Scale beta based on the ratio of current uncertainty to best value
    # This encourages more exploration when uncertainty is high relative to f_best
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # Dynamic beta: starts around 2.0, increases with uncertainty
    # Cap to avoid excessive exploration
    beta_base = 2.0
    uncertainty_ratio = sigma_mean / max(abs(f_best), 1e-9)
    beta = beta_base * (1.0 + 0.5 * np.tanh(uncertainty_ratio))
    beta = np.clip(beta, 1.0, 5.0)
    
    # LCB score: f_best - mu + beta * sigma
    lcb_score = f_best - mu + beta * sigma_safe
    
    # Combine EI and LCB:
    # Use a weighted combination that emphasizes EI for exploitation
    # and LCB for exploration, both normalized by cost
    # Weight: 0.6 for EI, 0.4 for LCB
    # This gives a balance between expected improvement and optimistic exploration
    
    # Normalize each component by its own scale to make them comparable
    ei_scale = np.max(ei) + 1e-9
    lcb_scale = np.max(lcb_score) + 1e-9
    
    ei_normalized = ei / ei_scale
    lcb_normalized = lcb_score / lcb_scale
    
    # Combine and normalize by cost
    combined = 0.6 * ei_normalized + 0.4 * lcb_normalized
    scores = combined / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
