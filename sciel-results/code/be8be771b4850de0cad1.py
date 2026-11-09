# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound acquisition for minimization.
    
    score = -mu + kappa * sigma - lambda * cost
    
    This directly optimizes for the minimum observed value (exploitation via -mu)
    while exploring uncertain regions (sigma) and penalizing expensive points (cost).
    The kappa parameter adapts based on the spread of sigma to balance exploration.
    """
    # Start with a deadline guard
    t_start = time.time()
    deadline = t_start + 290  # 10s safety margin before 300s limit
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma = np.clip(sigma, 1e-8, None)
    
    # Scale-aware kappa: use the median sigma as a reference scale
    # This makes kappa adapt to the problem's uncertainty scale
    sigma_median = np.median(sigma)
    
    # Adaptive kappa: larger when uncertainty is high relative to improvement
    # Base kappa around 2.5 (typical UCB value) with scaling
    # Use a formula that balances exploration and exploitation
    # kappa = 2.5 * sigma_median / (sigma_median + 1e-6)
    # But we want it to be scale-aware in objective units
    
    # For LCB in minimization: we want low mu (good) and high sigma (uncertain)
    # The cost term penalizes expensive evaluations
    
    # Normalize cost to [0, 1] range for the penalty term
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-9
    cost_norm = (cost - cost_min) / cost_range  # in [0, 1]
    
    # The penalty lambda should be in objective units
    # A reasonable lambda: we want the cost penalty to be comparable to typical improvements
    # Use the spread of mu as a scale reference
    mu_range = np.max(mu) - np.min(mu) + 1e-9
    
    # Lambda: scale the cost penalty to be a fraction of the objective range
    # This ensures the penalty is meaningful relative to the objective values
    lambda_cost = 0.1 * mu_range
    
    # Kappa: exploration coefficient
    # Use a value that scales with the uncertainty
    # Typical UCB uses beta = 2 * log(1 + t) / n or similar
    # Here we use a fixed but scale-aware kappa
    kappa = 2.0 + 0.5 * (sigma_median / (sigma_median + 0.1))
    
    # Compute the LCB score
    # Higher score = better (we want to maximize this)
    # -mu: lower mu is better (minimization)
    # +kappa * sigma: higher uncertainty is better (exploration)
    # -lambda * cost_norm: lower cost is better (efficiency)
    score = -mu + kappa * sigma - lambda_cost * cost_norm
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
# EVOLVE-BLOCK-END
