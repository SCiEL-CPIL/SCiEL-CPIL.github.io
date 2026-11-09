import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines LCB with dynamic budget-aware exploration scaling.
    """
    t0 = time.time()
    deadline = t0 + 250.0  # Safety margin before 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure no division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Budget-aware exploration scaling:
    # Remaining budget fraction drives how aggressively we explore
    # Total budget is 30, initial cost is 5, so remaining = 30 - cost
    # But cost here is per-point cost, not total. We need to estimate total cost.
    # Since we don't have total cost, we use the mean cost as a proxy for 
    # how "expensive" the current candidates are.
    
    # Use the mean cost to estimate where we are in the budget
    mean_cost = np.mean(cost_safe)
    
    # The total budget is 30. Initial points cost ~1.5 each (5 points * 1.5 = 7.5)
    # So remaining budget is roughly 30 - 7.5 = 22.5
    # But this is approximate. Let's use a simpler heuristic:
    # When mean_cost is low, we can afford more exploration
    # When mean_cost is high, we should exploit more
    
    # Normalize cost to [0, 1] range based on expected cost range [1, 2]
    cost_normalized = np.clip((cost_safe - 1.0) / 1.0, 0.0, 1.0)
    
    # Budget-aware beta: start high (exploration) and decrease as we "spend"
    # Use a decay based on the mean cost relative to the budget
    # Higher mean cost -> we've spent more -> lower beta
    base_beta = 2.5  # Initial exploration strength
    decay_factor = np.clip(mean_cost / 2.0, 0.0, 1.0)  # 0 to 1
    beta = base_beta * (1.0 - 0.7 * decay_factor)  # Decay from 2.5 to 0.75
    
    # Lower Confidence Bound for minimization
    # LCB = mu - beta * sigma
    # We want higher scores to be better, so we negate: -LCB = beta * sigma - mu
    lcb_score = beta * sigma_safe - mu
    
    # Also compute EI for exploitation
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Blend LCB and EI based on uncertainty
    # When sigma is high relative to the spread, favor LCB (exploration)
    # When sigma is low, favor EI (exploitation)
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean < 1e-9:
        sigma_mean = 1.0
    
    # Relative uncertainty
    rel_uncertainty = sigma_safe / (sigma_mean + 1e-9)
    
    # Weight for LCB: higher when uncertainty is high
    w_lcb = np.clip(0.5 + 0.5 * rel_uncertainty, 0.3, 0.8)
    w_ei = 1.0 - w_lcb
    
    # Combine
    raw_score = w_lcb * lcb_score + w_ei * ei
    
    # Cost-aware normalization
    scores = raw_score / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
