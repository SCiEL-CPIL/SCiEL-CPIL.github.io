"""Rewritten program with dynamic, instance-adaptive exploration budget."""
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function with dynamic exploration budget.
    
    Uses a hybrid approach:
    - Dynamic beta that scales with remaining budget (more exploration early, exploitation late)
    - Cost-aware normalization that accounts for the varying evaluation costs
    - Safety margin for numerical stability
    """
    # Track elapsed time for deadline guard
    t_start = time.time()
    DEADLINE = 300.0  # Must finish within 300s per evaluation
    
    # Input shapes: mu, sigma, cost all have shape (2000,)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check for numerical issues in inputs
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        # Replace non-finite values with safe defaults
        mu = np.nan_to_num(mu, nan=0.0, posinf=1e10, neginf=-1e10)
        sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
        cost = np.nan_to_num(cost, nan=1.0, posinf=10.0, neginf=1.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Dynamic exploration budget ---
    # Estimate remaining budget based on current cost landscape
    # The total budget is 30, initial 5 points cost ~5 * (1 + 0.5) = 7.5 on average
    # So remaining budget is roughly 30 - sum of past costs, but we don't have that directly.
    # Instead, we use the distribution of costs in the current batch as a proxy.
    
    # Median cost in the current candidate set
    cost_median = np.median(cost_safe)
    
    # Normalize costs to [0, 1] range for the current batch
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min + 1e-9
    cost_normalized = (cost_safe - cost_min) / cost_range
    
    # --- Adaptive beta based on uncertainty and budget ---
    # Key insight: when sigma is high relative to the spread of mu, we should explore more.
    # Also, as we approach the budget limit, we should exploit more.
    
    # Global uncertainty level
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    mu_range = np.ptp(mu) + 1e-9
    
    # Ratio of uncertainty to objective spread
    uncertainty_ratio = sigma_mean / (mu_range / 2 + 1e-9)
    
    # Base beta: higher when uncertainty is high relative to the objective landscape
    base_beta = 1.5 + 1.5 * min(uncertainty_ratio, 2.0)
    
    # Budget-aware scaling:
    # If average cost is low, we've likely spent less of our budget -> explore more
    # If average cost is high, we're closer to the limit -> exploit more
    # cost_normalized is in [0,1], so (1 - cost_normalized) is high when cost is low
    # But we want a single scalar multiplier, so use the median
    cost_median_normalized = (cost_median - cost_min) / cost_range
    
    # Budget pressure: 0 when cost is low (early), 1 when cost is high (late)
    budget_pressure = cost_median_normalized
    
    # Dynamic beta: start aggressive (high beta), taper off as budget pressure increases
    # beta_dynamic = base_beta * (1.0 - 0.5 * budget_pressure)
    # This gives beta from base_beta (early) to 0.5*base_beta (late)
    beta_dynamic = base_beta * (1.0 - 0.5 * budget_pressure)
    
    # Clamp beta to reasonable range
    beta_dynamic = np.clip(beta_dynamic, 0.5, 4.0)
    
    # --- Compute acquisition scores ---
    # For minimization: we want low mu and high sigma
    # UCB-style: score = -(mu - beta * sigma) / cost
    # = (beta * sigma - mu) / cost
    
    # Also incorporate an exploitation term: reward points near f_best
    # This helps when the GP is confident (low sigma) but mu is close to f_best
    mu_centered = mu - f_best  # Negative means below current best (good)
    
    # Hybrid: blend UCB with a pure exploitation signal
    # ucb_score = (beta_dynamic * sigma - mu) / cost_safe
    # exploit_score = -mu_centered / cost_safe  # Higher when mu is well below f_best
    # 
    # But mu_centered can be very negative if f_best is very good, which is fine.
    # Let's use a simpler approach: just UCB with dynamic beta.
    
    scores = (beta_dynamic * sigma - mu) / cost_safe
    
    # Add a small exploration bonus for regions with high sigma relative to local sigma
    # This encourages exploring the most uncertain regions
    if sigma_std > 1e-12:
        sigma_local = (sigma - sigma_mean) / (sigma_std + 1e-9)
        # Small bonus for high-sigma regions
        exploration_bonus = 0.1 * sigma_local / cost_safe
        scores = scores + exploration_bonus
    
    # Handle numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Ensure all scores are finite
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline guard: if we're running long, return immediately
    elapsed = time.time() - t_start
    if elapsed > DEADLINE - 5.0:
        # Safety margin: return what we have
        pass  # We're already done with computation, just returning
    
    return scores
