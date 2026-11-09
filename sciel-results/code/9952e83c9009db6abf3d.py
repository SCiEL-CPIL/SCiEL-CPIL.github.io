import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining UCB and EI for Bayesian optimization.
    
    Args:
        mu: (2000,) Mean of the GP predictions.
        sigma: (2000,) Standard deviation of the GP predictions.
        f_best: float, the best (lowest) value observed so far.
        cost: (2000,) Cost of evaluating each candidate.
        
    Returns:
        scores: (2000,) Acquisition scores. Higher scores indicate better candidates.
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -40, 40)
    
    # Expected Improvement for minimization
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # UCB for minimization: we want low mu + high sigma
    # UCB = mu - kappa * sigma (lower is better, so we negate for maximization)
    # Actually, for minimization, UCB score = -(mu - kappa*sigma) = -mu + kappa*sigma
    # But we want to maximize the acquisition, so let's think differently.
    # 
    # In minimization, a good point has low mu and high sigma (uncertainty).
    # UCB for minimization: score_ucb = -mu + kappa * sigma
    # But we also want to incorporate f_best relative improvement.
    #
    # Let's use a hybrid approach:
    # 1. EI captures expected improvement over f_best
    # 2. UCB captures exploration via uncertainty
    #
    # We'll blend them with a dynamic kappa that depends on remaining budget.
    
    # Estimate remaining budget. Total budget is 30, initial cost is 5 (5 uniform points).
    # We don't know exactly how much cost has been spent, but we can estimate.
    # The cost array gives us the cost of each candidate. The mean cost of evaluated points
    # would be around 1.5 (since mean(x) for uniform in [0,1]^3 is 0.5, so cost ~ 1.5).
    # Let's estimate total spent cost. We don't have direct access, but we can use
    # the fact that we're in a loop until cost < 30.
    #
    # A simpler approach: use a fixed kappa that balances exploration and exploitation.
    # For Hartmann 3D, a moderate kappa works well.
    
    # Dynamic kappa based on sigma distribution
    # When sigma is high overall, we should explore more
    # When sigma is low, we should exploit more
    
    # Compute a measure of overall uncertainty
    sigma_median = np.median(sigma_safe)
    sigma_mean = np.mean(sigma_safe)
    
    # Scale kappa based on how much uncertainty remains
    # If sigma is large relative to f_best, explore more
    # Normalize by typical scale of the problem (Hartmann 3D range is about 4-5)
    typical_scale = 1.0
    
    # kappa decreases as we become more certain
    # Base kappa around 1.5-2.5 for good exploration
    base_kappa = 2.0
    
    # Adjust kappa: if median sigma is small, reduce exploration
    # If median sigma is large, increase exploration
    kappa = base_kappa * np.clip(sigma_median / typical_scale, 0.3, 2.0)
    
    # UCB score for minimization (we want to maximize this)
    # ucb_score = -mu + kappa * sigma
    ucb_score = -mu + kappa * sigma_safe
    
    # Normalize EI and UCB to comparable scales
    # EI is in objective units, UCB is also in objective units
    # Blend them
    # Weight: more weight on EI when f_best is close to optimum, more on UCB when exploring
    
    # Simple blend: average of normalized versions
    # First, let's scale EI to be comparable
    ei_max = np.max(ei) if np.max(ei) > 1e-10 else 1.0
    ei_norm = ei / ei_max
    
    ucb_min = np.min(ucb_score) if np.min(ucb_score) < np.max(ucb_score) else 0.0
    ucb_range = np.max(ucb_score) - ucb_min
    if ucb_range < 1e-10:
        ucb_range = 1.0
    ucb_norm = (ucb_score - ucb_min) / ucb_range
    
    # Blend: 60% EI, 40% UCB
    blended = 0.6 * ei_norm + 0.4 * ucb_norm
    
    # Cost-aware: divide by cost
    cost_safe = np.maximum(cost, 1e-8)
    scores = blended / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard
    if time.time() - t_start > 280:
        pass
    
    return scores
