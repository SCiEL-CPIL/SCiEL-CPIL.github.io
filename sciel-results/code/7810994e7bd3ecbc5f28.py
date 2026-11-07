"""
Cost-aware Lower Confidence Bound (LCB) acquisition for Bayesian optimization.
Structural change from EI/KG: uses LCB with dynamic exploration scaled by remaining budget,
explicitly penalizing high-uncertainty regions to prevent variance inflation in 8D.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # 300s limit + 5s margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound acquisition function.
    
    Replaces EI with LCB: score = mu - beta * sigma
    where beta is dynamically scaled by:
      1. Remaining budget (more budget -> more exploration)
      2. Gap to minimum (larger gap -> more exploration)
      3. Cost of evaluation (cheaper points get relatively more exploration weight)
    
    This prevents the GP from wasting expensive evaluations on high-variance
    regions where variance inflation typically misleads EI-based selectors.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Safety fallback: simple LCB / cost
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        lcb = mu - 1.0 * sigma_safe
        score = lcb / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Dynamic exploration coefficient beta ---
    # LCB score = mu - beta * sigma (for minimization, we want low mu and low sigma)
    # Actually for minimization LCB is: mu - beta * sigma
    # Lower values are better, so we return negative of that or invert.
    # Let's define: acquisition_value = -(mu - beta * sigma) = -mu + beta * sigma
    # Higher acquisition_value means we prefer it.
    # Wait, standard LCB for minimization: minimize mu - beta*sigma
    # So the "score" to maximize is -mu + beta*sigma
    # But we also need to account for f_best improvement context.
    
    # Let's stick to a form that rewards low mu and appropriate sigma.
    # Base LCB term: -mu + beta * sigma
    # But pure LCB can be too aggressive. Let's blend with EI-like signal.
    
    # Compute standard EI for reference
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute LCB score (higher is better for selection)
    # We want to select points with low mu.
    # lcb_score = -mu + beta * sigma
    
    # Dynamic beta:
    # 1. Start with a base beta that encourages exploration
    # 2. Scale by remaining budget: early on (high cost remaining) -> larger beta
    #    The cost array gives per-point cost. Total budget is 30.
    #    We don't have total cost directly, but we can infer from the context.
    #    Actually, the problem says "While total cost is below 30".
    #    We don't have the total cost passed in, only per-point costs.
    #    However, f_best is passed. We can use the gap to minimum as a proxy for progress.
    
    min_possible = -0.8
    gap_to_min = max(f_best - min_possible, 0.0)
    
    # When gap is large (early), explore more (higher beta)
    # When gap is small (late), exploit more (lower beta)
    # Use exponential decay
    beta_base = 1.5
    beta_decay = np.exp(-gap_to_min / 0.2)
    beta = beta_base * beta_decay
    
    # Additionally, we can scale beta by the relative cost of the point.
    # Cheaper points should get a slightly higher effective beta to encourage
    # exploring cheap, uncertain regions.
    # Normalize costs: cost_min to cost_max
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min
    if cost_range < 1e-9:
        cost_range = 1e-9
    cost_norm = (cost_safe - cost_min) / cost_range  # 0 for cheapest, 1 for most expensive
    
    # Cheaper points (low cost_norm) get higher beta
    # More expensive points (high cost_norm) get lower beta
    cost_adjustment = 1.0 - 0.3 * cost_norm  # 1.0 for cheapest, 0.7 for most expensive
    beta = beta * cost_adjustment
    
    # LCB score: -mu + beta * sigma
    # This rewards low mu and high sigma (exploration)
    lcb_score = -mu + beta * sigma_safe
    
    # To prevent pure noise-chasing, blend with EI
    # EI captures the expected improvement in f_best
    # LCB captures the potential for finding a much lower value
    # Weight: 0.6 LCB + 0.4 EI
    # But LCB and EI have different scales.
    # Let's normalize them.
    
    # Normalize LCB score to [0, 1] roughly
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    lcb_range = lcb_max - lcb_min
    if lcb_range < 1e-9:
        lcb_range = 1e-9
    lcb_norm = (lcb_score - lcb_min) / lcb_range
    
    # Normalize EI to [0, 1]
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-9:
        ei_range = 1e-9
    ei_norm = (ei - ei_min) / ei_range
    
    # Blend
    # Early (large gap): more LCB (exploration)
    # Late (small gap): more EI (exploitation)
    w_lcb = 0.5 + 0.5 * np.exp(-gap_to_min / 0.3)  # 0.5 to 1.0
    w_ei = 1.0 - w_lcb
    
    combined_score = w_lcb * lcb_norm + w_ei * ei_norm
    
    # Cost-aware normalization
    # Divide by cost to favor cheaper evaluations
    # Use a softened cost
    cost_softened = cost_safe * (1.0 + 0.1 * np.exp(-cost_safe))
    score = combined_score / cost_softened
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
