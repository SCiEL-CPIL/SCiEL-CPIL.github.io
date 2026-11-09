"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Cost-Adjusted Lower Confidence Bound (LCB) with exploration term
    # score = mu + kappa * sigma - lambda * cost
    # We want to minimize f, so lower mu is better.
    # LCB for minimization: mu - kappa * sigma (we want low values)
    # But we also want to account for cost: subtract lambda * cost
    # So: score = mu - kappa * sigma - lambda * cost
    # Higher score means more likely to be a good (low) value with low cost
    
    # Ensure sigma is positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive kappa based on the relationship between f_best and the current mu
    # When f_best is much lower than mu, we should explore less (or more?)
    # Actually, standard LCB uses a fixed kappa, but we can make it adaptive
    # Let's use a simple approach: kappa that balances exploitation and exploration
    
    # The cost term: cost is in [1, 2] (since 1 + mean(x scaled to [0,1]) and x in [0,1]^2)
    # Actually, cost = 1 + mean(x_scaled) where x_scaled is in [0,1] for each coordinate
    # So cost is in [1, 2]
    
    # Let's use a fixed kappa and lambda, tuned for this problem
    # Since we're minimizing, we want low mu and low cost
    # score = mu - kappa * sigma - lambda * cost
    
    # But we need to be careful: if mu is very high (far from f_best), we might want to explore
    # Let's use a form that emphasizes finding new minima
    
    # Alternative: use a normalized form
    # z = (f_best - mu) / sigma  (positive when mu < f_best, i.e., potential improvement)
    # But for LCB, we just use mu - kappa * sigma
    
    # Let's try: score = mu - 1.0 * sigma - 0.1 * cost
    # This encourages low mu, high uncertainty (exploration), and low cost
    
    kappa = 1.0
    lambda_cost = 0.1
    
    score = mu - kappa * sigma_safe - lambda_cost * cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
