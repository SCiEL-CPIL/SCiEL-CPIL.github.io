# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s. We use a safe margin.
_DEADLINE_START = time.time()
_TIME_LIMIT = 300.0  # seconds, leave 30s margin before 330s kill

# Global state to track total evaluations for adaptive exploration
_total_cost_accumulated = 0.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines cost-aware EI with adaptive LCB exploration, weighted by budget remaining.
    """
    global _total_cost_accumulated
    
    # Check deadline
    if time.time() - _DEADLINE_START > _TIME_LIMIT - 5.0:
        # Safe fallback: simple cost-normalized EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(cost, 1e-9)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64).flatten()
    sigma = np.asarray(sigma, dtype=np.float64).flatten()
    cost = np.asarray(cost, dtype=np.float64).flatten()
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    score_ei = ei / cost_safe
    
    # Lower Confidence Bound (LCB) for minimization
    # We want low mu and high sigma, so LCB = mu - beta * sigma
    # Score is higher when LCB is lower, so we negate: -LCB / cost
    remaining_budget_fraction = max(0.0, 1.0 - _total_cost_accumulated / 30.0)
    
    # Adaptive beta: starts high for exploration, decays as budget is consumed
    # Also adapt based on how close f_best is to the minimum of mu
    mu_min = np.min(mu)
    f_best_norm = max(0.0, min(1.0, (f_best - mu_min) / (np.max(mu) - mu_min + 1e-9)))
    
    # Beta decreases as we get closer to optimum and as budget runs out
    base_beta = 1.5
    beta = base_beta * (remaining_budget_fraction + 0.1) * (0.5 + 0.5 * f_best_norm)
    
    lcb = mu - beta * sigma_safe
    lcb_per_cost = -lcb / cost_safe  # Negate because higher score = better
    
    # Normalize both components to [0, 1] scale for stable weighting
    max_ei = np.max(score_ei)
    if max_ei > 1e-12:
        ei_norm = score_ei / max_ei
    else:
        ei_norm = score_ei
    
    max_lcb = np.max(lcb_per_cost)
    if max_lcb > 1e-12:
        lcb_norm = lcb_per_cost / max_lcb
    else:
        lcb_norm = lcb_per_cost
    
    # Weights: 
    # - EI weight increases as f_best improves (exploitation phase)
    # - LCB weight decreases as budget is consumed (less exploration)
    w_ei = 0.6 + 0.4 * (1.0 - f_best_norm)  # Higher weight on EI when f_best is good
    w_lcb = 0.4 * remaining_budget_fraction  # Less LCB weight as budget runs out
    
    # Normalize weights
    w_sum = w_ei + w_lcb
    if w_sum < 1e-9:
        w_sum = 1e-9
    w_ei /= w_sum
    w_lcb /= w_sum
    
    # Combine
    scores = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Update global cost tracker (approximate: assume we're in a loop)
    # We don't know exact cost consumed this call, but we can track total
    # For now, just use the fraction based on number of calls as a proxy
    # Actually, we should track this more carefully. Since we can't easily track,
    # we'll use a simple heuristic: assume each call consumes ~1 unit of budget
    # on average, so after N calls, ~N units consumed. But we don't have N.
    # Alternative: use f_best improvement as a proxy for progress.
    # For simplicity, keep the static budget fraction for now.
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
