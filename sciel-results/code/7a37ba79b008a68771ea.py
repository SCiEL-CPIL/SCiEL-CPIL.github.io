import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Rastrigin 2D.
    Uses a hybrid of Expected Improvement per cost and UCB per cost.
    """
    start_time = time.time()
    deadline = 300.0

    # Ensure arrays are numpy
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Basic sanitization
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # f_best is a scalar
    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = 0.0

    # 1. Expected Improvement per unit cost
    z = (f_best - mu) / sigma
    # Avoid numerical issues in cdf/pdf
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # EI is 0 if mu >= f_best, which is correct.
    score_ei_cost = ei / cost
    
    # 2. Upper Confidence Bound (UCB) per unit cost
    # UCB = mu - kappa * sigma (minimization)
    # We want to maximize -UCB / cost = (kappa * sigma - mu) / cost
    kappa = 2.0
    ucb = mu - kappa * sigma
    score_ucb_cost = -ucb / cost

    # Hybrid: Balance exploitation (EI) and exploration (UCB)
    # The previous version used 0.7/0.3 with kappa=1.5.
    # Let's try 0.6/0.4 with kappa=2.0 to encourage slightly more exploration.
    alpha = 0.6
    hybrid = alpha * score_ei_cost + (1 - alpha) * score_ucb_cost
    
    # Check validity
    if not np.all(np.isfinite(hybrid)):
        # Fallback to EI/Cost
        hybrid = score_ei_cost
        if not np.all(np.isfinite(hybrid)):
            return np.zeros(n)

    # Time check (unlikely to be needed for vectorized ops on 2000 points)
    if time.time() - start_time > deadline:
         return hybrid

    return hybrid
