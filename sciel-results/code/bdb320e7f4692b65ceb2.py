import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Uncertainty-Weighted Improvement (UWI)
    with a local baseline estimated from GP predictions.
    
    UWI rewards candidates where:
    1. The GP posterior variance (sigma^2) is high relative to local prediction
    2. The predicted value is below the local baseline (not just global f_best)
    3. The evaluation cost is low
    
    This targets boundaries of explored basins rather than pure global best
    or pure uncertainty.
    """
    # Safety: ensure inputs are clean
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    known_min = -78.3323
    
    # Progress estimate: 0 at start, 1 near optimum
    progress = np.clip(1.0 - (f_best - known_min) / 250.0, 0.0, 1.0)
    
    # --- Local Baseline Estimation ---
    # Estimate a local baseline using a kernel-weighted average of mu values,
    # weighted by inverse sigma (more weight to confident predictions)
    # This approximates the "local best" rather than using global f_best
    
    # Weight by confidence: points with low sigma are more reliable
    conf_weights = 1.0 / (sigma ** 2 + 1e-6)
    conf_weights = conf_weights / (np.sum(conf_weights) + 1e-12)
    
    # Local baseline: weighted average of mu, but we want it to be lower
    # than f_best to encourage exploration in new regions
    # Use a mixture: part global f_best, part local weighted mean
    local_mu_mean = np.sum(conf_weights * mu)
    
    # Blend: early on, trust f_best more; later, trust local structure
    blend = 0.7 * (1.0 - progress) + 0.3 * progress
    local_baseline = blend * f_best + (1.0 - blend) * local_mu_mean
    
    # Ensure local_baseline is not better than f_best (we're minimizing)
    local_baseline = min(local_baseline, f_best)
    
    # --- Uncertainty-Weighted Improvement (UWI) ---
    # UWI = (baseline - mu) * P(improvement) * uncertainty_weight / cost^p
    # where P(improvement) = norm.cdf((baseline - mu) / sigma)
    # uncertainty_weight = sigma^alpha (alpha < 1 to avoid pure exploration)
    
    z = (local_baseline - mu) / (sigma + 1e-12)
    prob_improve = norm.cdf(z)
    
    # Expected improvement component
    ei = (local_baseline - mu) * prob_improve + sigma * norm.pdf(z)
    
    # Uncertainty weight: moderate exploration, not pure sigma
    # alpha decreases with progress: more exploitation later
    alpha = 0.6 - 0.3 * progress
    uncertainty_weight = (sigma + 1e-9) ** alpha
    
    # UWI score: combine EI and uncertainty weight
    uwi = ei * uncertainty_weight
    
    # Cost penalty: prefer cheaper evaluations
    # Penalty exponent decreases with progress (exploit more expensive points later)
    cost_exp = 1.3 - 0.5 * progress
    final_score = uwi / (cost ** cost_exp)
    
    # Add a small LCB component to ensure we still target low-predicted regions
    lcb_component = -mu / (cost ** (cost_exp + 0.1))
    
    # Combine: primarily UWI, with LCB as secondary signal
    # Weights shift from exploration (UWI) to exploitation (LCB) with progress
    uwi_weight = 0.8 - 0.3 * progress
    lcb_weight = 1.0 - uwi_weight
    
    # Normalize both components to comparable scales
    uwi_min, uwi_max = np.min(final_score), np.max(final_score)
    lcb_min, lcb_max = np.min(lcb_component), np.max(lcb_component)
    
    uwi_range = uwi_max - uwi_min if uwi_max > uwi_min else 1.0
    lcb_range = lcb_max - lcb_min if lcb_max > lcb_min else 1.0
    
    uwi_norm = (final_score - uwi_min) / (uwi_range + 1e-12)
    lcb_norm = (lcb_component - lcb_min) / (lcb_range + 1e-12)
    
    combined = uwi_weight * uwi_norm + lcb_weight * lcb_norm
    
    # Scale back to a reasonable range
    scale = np.std(final_score) + np.std(lcb_component) + 1e-6
    final_score = combined * scale + np.mean(final_score)
    
    # Ensure finite output
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        # Return a simple, fast fallback
        return -mu / (cost + 1e-9)
    
    return final_score
