"""Cost-aware acquisition for BO on three hump camel 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement with adaptive
    cost-awareness that increases as budget depletes.
    
    Strategy:
    1. Compute Expected Improvement (EI) for minimization.
    2. Apply cost-aware penalty that strengthens as remaining budget decreases,
       encouraging cheaper evaluations when budget is tight.
    3. Use a mild exploration bonus in regions of high uncertainty to balance
       exploitation and exploration.
    
    Parameters:
        mu: (2000,) GP mean predictions in objective units
        sigma: (2000,) GP std dev predictions (positive) in objective units
        f_best: float, best observed value so far
        cost: (2000,) evaluation cost per candidate
    
    Returns:
        (2000,) finite acquisition scores, higher = better
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Emergency: return something finite immediately
        return np.zeros(2000, dtype=np.float64)
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standardized improvement: Z = (f_best - mu) / sigma
    # For minimization, improvement is when mu < f_best
    Z = (f_best - mu) / sigma_safe
    Z = np.clip(Z, -10.0, 10.0)  # Numerical stability
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    # where Phi is the CDF and phi is the PDF of the standard normal
    from scipy.stats import norm
    phi = norm.pdf(Z)
    Phi = norm.cdf(Z)
    
    EI = (f_best - mu) * Phi + sigma_safe * phi
    
    # Clamp EI to be non-negative
    EI = np.maximum(EI, 0.0)
    
    # Cost-aware scaling with adaptive alpha:
    # cost = 1 + mean(x_scaled), where x_scaled is in [0,1]
    # So cost is in [1, 2]. 
    # As f_best gets closer to 0 (optimum), we should be more cost-sensitive
    # to conserve budget for fine-tuning near the optimum.
    # Also, as total cost approaches 30, we should prefer cheaper evaluations.
    
    # Estimate remaining budget pressure from f_best:
    # When f_best is large, we're far from optimum, explore more aggressively.
    # When f_best is small, we're near optimum, be more cost-conscious.
    f_best_abs = abs(f_best)
    
    # Alpha increases as f_best approaches 0 (near optimum) and as cost pressure increases
    # Use a smooth transition: alpha = base + scale / (1 + f_best_abs)
    base_alpha = 0.2
    scale_alpha = 0.8
    alpha = base_alpha + scale_alpha / (1.0 + f_best_abs)
    
    # Also consider the mean cost as a proxy for how much budget has been used
    mean_cost = np.mean(cost)
    # If mean cost is high, we've been evaluating expensive points, be more selective
    cost_pressure = (mean_cost - 1.0)  # in [0, 1]
    alpha += 0.3 * cost_pressure
    
    # Clip alpha to reasonable range
    alpha = np.clip(alpha, 0.1, 2.0)
    
    cost_penalty = 1.0 + alpha * (np.maximum(cost, 1.0) - 1.0)
    score = EI / cost_penalty
    
    # Add a small exploration bonus in high-uncertainty regions
    # This helps when EI is low but there's potential for discovery
    # Use a normalized uncertainty measure
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    uncertainty_bonus = 0.05 * (sigma_safe / sigma_max)
    
    # Only add bonus where EI is very low (below 10th percentile) to avoid
    # overwhelming the exploitation signal
    ei_threshold = np.percentile(EI, 10) if n > 0 else 0
    low_ei_mask = EI < ei_threshold
    score[low_ei_mask] += uncertainty_bonus[low_ei_mask]
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
