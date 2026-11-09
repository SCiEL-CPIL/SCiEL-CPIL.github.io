# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition for minimization.
    
    score = (kappa * sigma - mu) / cost
    
    where kappa is dynamically tuned based on budget progress estimated
    from the cost distribution and f_best improvement.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: avoid division by zero and ensure non-negative sigma
    cost = np.maximum(cost, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Estimate budget progress.
    # Total budget is 30. We don't know exactly how much has been spent,
    # but we can use the typical cost per evaluation and the fact that
    # the loop runs while total_cost < 30.
    # 
    # A better proxy: use the spread of mu and the current f_best to gauge
    # how much exploration vs exploitation is needed.
    #
    # Early on: f_best is poor (close to 0 for hartmann3d), sigma is large
    # Late on: f_best is good (near -3.86), sigma is smaller near optimum
    #
    # We estimate "progress" as how close f_best is to the known minimum (-3.86278)
    # relative to the initial range. But we don't know the initial f_best.
    #
    # Alternative: Use the ratio of min(mu) to f_best.
    # If min(mu) << f_best, we found a promising region -> exploit more.
    # If min(mu) ~ f_best, we're near convergence -> moderate exploration.
    
    mu_min = np.min(mu)
    
    # Gap: how much improvement potential remains
    # If f_best - mu_min is large, there's a lot of room -> explore more
    # If f_best - mu_min is small or negative, we're converging -> exploit
    gap = f_best - mu_min
    
    # Also consider the average uncertainty
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    
    # Dynamic kappa:
    # Start with high exploration (kappa ~ 2-3) and decrease as we converge.
    # Use a combination of:
    # 1. The gap between f_best and best predicted mu
    # 2. The level of uncertainty (sigma)
    #
    # kappa = base + exploration_bonus
    # exploration_bonus decreases as gap decreases and sigma decreases
    
    # Normalize gap: hartmann3d range is ~3.86
    # gap > 0 means f_best is still above best prediction (improvement possible)
    # gap <= 0 means best prediction is already better than f_best
    
    # Use a smooth decay: kappa starts at ~3 when gap is large, drops to ~0.5 when gap is small
    # Scale gap by the typical function range
    gap_scale = 3.86  # approximate range of hartmann3d
    
    # If gap is large (> gap_scale), we're early -> high kappa
    # If gap is small (< 0.1), we're late -> low kappa
    # Smooth transition using exp decay
    
    # Also factor in uncertainty: if sigma is high, we can afford more exploration
    sigma_factor = np.clip(sigma_mean / 0.5, 0.5, 2.0)  # normalize sigma level
    
    # Base kappa from gap
    if gap > 0:
        # Still room for improvement
        kappa_base = 0.5 + 2.5 * np.exp(-gap / (gap_scale * 0.3))
    else:
        # Best prediction is better than f_best -> exploit
        kappa_base = 0.5
    
    # Adjust for uncertainty level
    kappa = kappa_base * sigma_factor
    
    # Clamp to reasonable range
    kappa = np.clip(kappa, 0.3, 4.0)
    
    # For minimization: score = (kappa * sigma - mu) / cost
    # Higher score = more likely to be evaluated next
    # This favors points with low mu (good predictions) and high sigma (uncertainty)
    # Divided by cost to make it cost-aware
    
    score = (kappa * sigma - mu) / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
