"""Cost-aware acquisition for Bayesian optimization on rastrigin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE = time.time() + 320.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Quantile-Expected Improvement (QEI) acquisition with adaptive
    exploration for Bayesian optimization of rastrigin 2d.
    
    QEI = E[max(0, f_best - f) | f < f_best - q] * P(f < f_best - q)
    where q is an adaptive quantile threshold.
    
    Combined with a UCB exploration term and cost-aware normalization.
    
    Parameters:
        mu: (2000,) GP mean in objective units
        sigma: (2000,) GP standard deviation in objective units
        f_best: lowest observed value
        cost: (2000,) cost of evaluating each candidate
    
    Returns:
        (2000,) finite scores, higher = evaluate next
    """
    start_time = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)
    
    # --- QEI (Quantile Expected Improvement) ---
    # QEI maximizes the expected improvement weighted by the probability
    # of achieving at least a quantile-level improvement.
    # For a normal(mu, sigma^2), the QEI at quantile alpha is:
    #   QEI = (f_best - mu) * Phi(z_q) + sigma * phi(z_q) - q_alpha * Phi(z_q)
    # where z_q = (f_best - mu - q_alpha) / sigma, q_alpha = alpha * sigma
    # Simplified: use a fixed quantile level that adapts to f_best
    
    # Adaptive quantile level: more aggressive when f_best is large
    if f_best > 20.0:
        quantile = 0.3  # explore more, lower bar
    elif f_best > 5.0:
        quantile = 0.5
    elif f_best > 1.0:
        quantile = 0.7
    else:
        quantile = 0.9  # exploit, high bar
    
    q_alpha = quantile * sigma  # adaptive threshold
    
    # z for QEI: (f_best - mu - q_alpha) / sigma
    z_qei = (f_best - mu - q_alpha) / sigma
    z_qei = np.clip(z_qei, -10, 10)
    
    # QEI = (f_best - mu - q_alpha) * Phi(z) + sigma * phi(z)
    # This is the expected excess improvement above the quantile threshold
    qei = (f_best - mu - q_alpha) * norm.cdf(z_qei) + sigma * norm.pdf(z_qei)
    qei = np.maximum(qei, 0.0)
    
    # --- UCB exploration term ---
    # Adaptive beta based on uncertainty and f_best
    mean_sigma = np.mean(sigma)
    sigma_std = np.std(sigma)
    
    # Beta increases with overall uncertainty and when f_best is far from optimum
    base_beta = 1.5
    if f_best > 50.0:
        base_beta = 2.2
    elif f_best > 10.0:
        base_beta = 1.8
    elif f_best > 1.0:
        base_beta = 1.3
    else:
        base_beta = 0.8
    
    # Boost exploration when there's high variance in uncertainty
    uncertainty_boost = 0.5 * (sigma_std / (mean_sigma + 1e-8))
    beta = base_beta + uncertainty_boost
    
    ucb = -mu + beta * sigma
    
    # --- Adaptive weighting between QEI and UCB ---
    if f_best < 0.5:
        qei_weight = 0.80
        ucb_weight = 0.20
    elif f_best < 2.0:
        qei_weight = 0.70
        ucb_weight = 0.30
    elif f_best < 10.0:
        qei_weight = 0.55
        ucb_weight = 0.45
    else:
        qei_weight = 0.40
        ucb_weight = 0.60
    
    # Normalize to comparable scales
    qei_max = np.max(qei)
    if qei_max < 1e-10:
        qei_normalized = np.zeros_like(qei)
    else:
        qei_normalized = qei / (qei_max + 1e-10)
    
    ucb_min = np.min(ucb)
    ucb_range = np.max(ucb) - ucb_min
    if ucb_range < 1e-10:
        ucb_normalized = np.zeros_like(ucb)
    else:
        ucb_normalized = (ucb - ucb_min) / (ucb_range + 1e-10)
    
    # Combine
    score = qei_weight * qei_normalized + ucb_weight * ucb_normalized
    
    # --- Cost-awareness: sqrt(cost) normalization ---
    # Cost ranges from 1.0 to ~2.0, sqrt gives smoother scaling
    cost_norm = np.sqrt(cost)
    score = score / cost_norm
    
    # --- Small Thompson-sampling-style exploration bonus ---
    # Add a small bonus proportional to sigma to encourage visiting
    # high-uncertainty regions, scaled by how far we are from optimum
    if f_best > 1.0:
        ts_bonus = 0.05 * sigma / (1.0 + sigma)
    else:
        ts_bonus = 0.02 * sigma / (1.0 + sigma)
    score = score + ts_bonus / cost_norm
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 280.0:
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
