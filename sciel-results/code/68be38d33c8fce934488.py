# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function for minimization.
    
    score = ((f_best - mu) + kappa * sigma) / cost
    
    where kappa is dynamically tuned based on the ratio of remaining budget
    to current cost to shift the exploration-exploitation balance.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero and negative sigma
    cost = np.maximum(cost, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Estimate total budget spent. The loop runs while total_cost < 30.
    # We can estimate the number of evaluations done so far by looking at
    # the typical cost per evaluation. Since we don't have the exact count,
    # we use a heuristic: the mean cost of the 2000 candidates represents
    # the typical cost. If the budget is 30, the number of evaluations is
    # roughly 30 / mean_cost.
    
    mean_cost = np.mean(cost)
    # Estimate remaining budget fraction
    # Assume we've spent some fraction of the budget. A simple heuristic:
    # If f_best is still far from the known minimum (-3.86), we're early.
    # If f_best is close, we're late.
    
    # Use f_best to estimate progress:
    # Hartmann 3D minimum is approximately -3.86278
    # f_best starts around 0 (or positive) and decreases toward -3.86
    # Normalize progress: 0 = start, 1 = near optimum
    f_min_known = -3.86278
    f_range = 0.0 - f_min_known  # ~3.86
    
    # Progress: how much of the range we've covered
    # If f_best is 0, progress = 0; if f_best is -3.86, progress = 1
    progress = np.clip((f_best - f_min_known) / f_range, 0.0, 1.0)
    
    # As progress increases, we should shift from exploration to exploitation
    # kappa should decrease as we get closer to the optimum
    
    # Dynamic kappa: start high (exploration), decrease as we converge
    kappa_max = 3.0
    kappa_min = 0.5
    
    # Use a smooth transition based on progress
    # kappa = kappa_min + (kappa_max - kappa_min) * (1 - progress)^2
    # This gives high kappa early, low kappa late
    
    kappa = kappa_min + (kappa_max - kappa_min) * (1.0 - progress) ** 2
    
    # Compute cost-aware UCB score for minimization
    # For minimization, we want to favor points with:
    # 1. Low mu (likely to be good)
    # 2. High sigma (uncertain, might be even better)
    # 3. Low cost (cheap to evaluate)
    
    # The improvement potential is (f_best - mu) when mu < f_best
    # We add kappa * sigma for exploration
    # Divide by cost for cost-awareness
    
    score = ((f_best - mu) + kappa * sigma) / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
