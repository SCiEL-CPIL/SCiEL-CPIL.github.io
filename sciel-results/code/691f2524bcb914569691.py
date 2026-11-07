"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Probability of Improvement (PoI)
    with an adaptive exploration bonus, divided by cost.
    
    This approach uses a structural blend:
    1. PoI as the primary exploitation signal (robust for multimodal Rastrigin)
    2. A cost-effective uncertainty term as the exploration signal
    3. A power-mean combination that adapts the balance based on f_best
    4. Final division by cost for cost-efficiency
    
    The power-mean combination (inspired by Pareto-frontier thinking) preserves
    both signals without letting cost arbitrarily dominate, while the PoI base
    (from the top-scoring inspiration) provides robust probabilistic guidance.
    
    Args:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP standard deviation
        f_best: float - best observed value
        cost: array of shape (2000,) - cost of evaluating each point
        
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: simple cost-aware PoI
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (float(f_best) - mu) / sigma_safe
        poi = norm.cdf(z)
        return poi / (cost + 1e-9)
    
    # Ensure inputs are properly typed
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle f_best
    if hasattr(f_best, 'min'):
        f_best_val = float(np.min(f_best))
    else:
        f_best_val = float(f_best)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # === Metric 1: Probability of Improvement (primary exploitation signal) ===
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best_val - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # === Metric 2: Cost-effective uncertainty (exploration signal) ===
    # Normalized sigma divided by cost: how much uncertainty we get per unit cost
    sigma_range = np.percentile(sigma, 95) - np.percentile(sigma, 5)
    if sigma_range < 1e-12:
        sigma_range = 1.0
    unc_norm = (sigma - np.percentile(sigma, 5)) / sigma_range
    unc_norm = np.clip(unc_norm, 0.0, 1.0)
    
    # Cost-effective uncertainty: higher sigma, lower cost = higher score
    cost_safe = np.maximum(cost, 1e-9)
    cost_eff_unc = unc_norm / cost_safe
    
    # Normalize cost_eff_unc to [0, 1] using robust percentiles
    lo_c = np.percentile(cost_eff_unc, 5)
    hi_c = np.percentile(cost_eff_unc, 95)
    if hi_c - lo_c < 1e-12:
        cost_eff_unc_norm = np.zeros_like(cost_eff_unc)
    else:
        cost_eff_unc_norm = (cost_eff_unc - lo_c) / (hi_c - lo_c)
        cost_eff_unc_norm = np.clip(cost_eff_unc_norm, 0.0, 1.0)
    
    # === Adaptive alpha: balance exploitation vs exploration ===
    # Early (high f_best): explore more (lower alpha on PoI, higher on uncertainty)
    # Late (low f_best): exploit more (higher alpha on PoI)
    # f_best for Rastrigin ranges from ~200 (random) to 0 (minimum)
    alpha = 0.5 + 0.35 * np.tanh(f_best_val / 15.0)
    alpha = np.clip(alpha, 0.15, 0.95)
    
    # === Power-mean combination ===
    # This preserves both signals structurally:
    # - If both are high, score is high
    # - If one is very low, the power mean penalizes appropriately
    # - Unlike weighted sum, this prevents one metric from dominating
    poi_safe = np.maximum(poi, 1e-12)
    unc_safe = np.maximum(cost_eff_unc_norm, 1e-12)
    
    combined = (np.power(poi_safe, alpha) + np.power(unc_safe, 1.0 - alpha)) / 2.0
    
    # === Cost-aware final score ===
    # Divide by cost to get "improvement per unit cost"
    scores = combined / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
