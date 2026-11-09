import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on powell 4d.
    Uses a dynamic exploration budget that scales with remaining budget.
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Dynamic exploration weight:
    # - Use f_best to gauge how far we are from the optimum (0 for Powell 4d)
    # - Scale exploration based on remaining budget
    #   Budget is 30, initial 5 points cost ~7.5, so remaining ~22.5
    #   We don't know exact current cost, but can infer from f_best
    #   Early on (high f_best): explore more
    #   Late (low f_best): exploit more
    
    # Estimate remaining budget fraction
    # f_best starts high and decreases toward 0
    # Use a sigmoid-like function to map f_best to exploration weight
    # When f_best is large (early), beta is high; when small (late), beta is low
    
    # Typical f_best range: starts around 10-20, ends near 0
    # Use beta = clip(a * f_best + b, min_beta, max_beta)
    
    # From the inspiration: beta = clip(0.5 * f_best, 0.1, 2.0)
    # But let's make it more dynamic based on budget considerations
    
    # The cost structure: cost = 1 + mean(x_scaled)
    # Mean cost per evaluation is ~1.5
    # Total budget 30, so ~20 evaluations total
    # 5 initial, so ~15 more
    
    # Use a decay schedule based on f_best
    # When f_best > 5: aggressive exploration (beta ~ 1.5-2.0)
    # When f_best < 1: precise exploitation (beta ~ 0.1-0.3)
    # In between: interpolate
    
    # Use a smooth transition
    beta_base = np.clip(0.3 * f_best + 0.2, 0.1, 2.5)
    
    # Additional cost-awareness:
    # Penalize high-cost points more when budget is tight
    # But we don't know exact remaining budget, so use a fixed cost penalty
    
    # Primary score: (EI + beta * sigma) / cost
    # This naturally balances exploration and exploitation while accounting for cost
    
    score = (ei + beta_base * sigma_safe) / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > 280.0:  # Safety margin well before 300s limit
        # Return a simpler, faster score
        simple_score = ei / cost
        simple_score = np.where(np.isfinite(simple_score), simple_score, 0.0)
        simple_score = np.nan_to_num(simple_score, nan=0.0, posinf=0.0, neginf=0.0)
        return simple_score
    
    return score
