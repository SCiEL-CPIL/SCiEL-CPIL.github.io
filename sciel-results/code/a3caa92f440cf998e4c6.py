"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import erf


def _norm_cdf(x):
    """Vectorized standard normal CDF using erf."""
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    
    # Clip to avoid overflow/underflow issues in erf, though erf is stable, 
    # we cap the probability at 0 and 1 for logical clarity.
    mask_large_pos = x > 8.0
    mask_large_neg = x < -8.0
    mask_mid = ~(mask_large_pos | mask_large_neg)
    
    result[mask_large_pos] = 1.0
    result[mask_large_neg] = 0.0
    
    if np.any(mask_mid):
        xm = x[mask_mid]
        result[mask_mid] = 0.5 * (1.0 + erf(xm / np.sqrt(2.0)))
    
    return result


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for minimizing shekel 4d.
    
    Strategy:
    1. Primary signal: Probability of Improvement (POI).
       This focuses on candidates that are highly likely to beat the current best.
       This was identified as a superior structural choice for this task compared to EI.
    2. Secondary signal: Cost-aware Exploration (LCB).
       A small LCB term is added to encourage exploration in high-uncertainty, low-cost regions.
    3. Cost Adjustment: The combined score is divided by a power of the cost to penalize expensive evaluations.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions (in objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions (in objective units).
    f_best : float
        Lowest observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    
    # Safety deadline guard: leave margin before 330s wall-clock
    DEADLINE = 320.0  # seconds
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Component 1: Probability of Improvement (POI) ---
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This is the primary driver. High POI means high confidence in improvement.
    delta = f_best - mu
    z_poi = delta / sigma_safe
    # Clip z to avoid numerical issues, though _norm_cdf handles extremes
    z_poi_clipped = np.clip(z_poi, -50, 50)
    poi = _norm_cdf(z_poi_clipped)
    poi = np.maximum(poi, 0.0)
    
    # --- Component 2: Cost-Aware Exploration (LCB) ---
    # We use LCB to explore, but we want to suppress it in high-cost areas.
    # LCB(x) = mu - kappa * sigma
    # We want to maximize -LCB.
    # Instead of a global kappa, we use a relative exploration term.
    # Exploration is valuable when sigma is high.
    # We scale the exploration bonus by 1/cost to favor cheap exploration.
    
    # Base exploration weight
    # If we use pure LCB, it might explore too much. 
    # We blend it with POI.
    # Let's define an exploration score: E(x) = -mu + kappa * sigma
    # We want to normalize this to be comparable to POI (which is in [0,1]).
    
    # Calculate LCB values
    # Kappa is chosen to be moderate. Since we are adding this to POI, 
    # we need to scale it down significantly so POI remains dominant.
    kappa = 1.0
    lcb_vals = mu - kappa * sigma_safe
    
    # To make LCB comparable to POI (0-1), we normalize LCB.
    # Lower LCB is better. 
    # lcb_min is the most promising (lowest) value.
    # We map [lcb_min, lcb_max] to [1, 0] so that lower LCB gets higher score.
    lcb_min = np.min(lcb_vals)
    lcb_max = np.max(lcb_vals)
    lcb_range = lcb_max - lcb_min if lcb_max > lcb_min else 1.0
    lcb_norm = 1.0 - (lcb_vals - lcb_min) / lcb_range
    lcb_norm = np.clip(lcb_norm, 0.0, 1.0)
    
    # --- Blending ---
    # The cell finding suggests POI is much stronger. 
    # We will use a heavy weight on POI and a small weight on LCB.
    # Weight of POI: 0.9
    # Weight of LCB: 0.1
    
    # However, we also need to consider that LCB is only useful if there is uncertainty.
    # If sigma is very small, LCB ~ mu, which is similar to exploitation.
    # POI already handles exploitation well.
    
    # Let's refine the blend:
    # Score = w_poi * POI + w_lcb * LCB_norm
    # But LCB_norm is a relative rank. 
    # A better approach: 
    # Use POI as the main score.
    # Add a small bonus for exploration in cheap, uncertain regions.
    
    # Exploration Bonus: 
    # High when sigma is high and cost is low.
    # sigma_max = np.max(sigma_safe)
    # sigma_rel = sigma_safe / (sigma_max + 1e-8)
    # cost_min = np.min(cost_safe)
    # cost_max = np.max(cost_safe)
    # cost_rel = (cost_safe - cost_min) / (cost_max - cost_min + 1e-8)
    # exploration_bonus = sigma_rel * (1 - cost_rel)
    
    # Let's stick to the LCB norm as it captures the mu-sigma tradeoff directly.
    # We will weight it low.
    
    w_poi = 0.95
    w_lcb = 0.05
    
    combined = w_poi * poi + w_lcb * lcb_norm
    
    # --- Cost Adjustment ---
    # The cost is 1 + mean(x). 
    # We want to divide by cost to penalize expensive points.
    # Using a power of 0.5 (sqrt) is a common heuristic for cost-awareness 
    # to not be too aggressive.
    # scores = combined / np.sqrt(cost_safe)
    
    # Let's try power 0.7 as in the current program, which seemed reasonable, 
    # or perhaps 1.0 (linear) to strongly favor cheap points.
    # Given the cost range is [1, 2] (since x in [0,10], mean(x) in [0,1], 
    # wait, x is in [0,10]^4. 
    # The prompt says: "Evaluating x costs 1 + mean(x scaled to [0, 1])."
    # So x_scaled is in [0,1]. mean(x_scaled) is in [0,1].
    # Cost is in [1, 2].
    # The variation in cost is small (factor of 2).
    # Therefore, a strong cost penalty might not be necessary or might distort the POI signal too much.
    # A gentle penalty is better.
    
    scores = combined / np.power(cost_safe, 0.5)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    if time.time() - t0 > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
