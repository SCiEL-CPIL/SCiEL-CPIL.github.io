"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_DEADLINE = 330.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PI) acquisition function.
    
    Strategy:
    1. Compute Probability of Improvement (PI) for minimization:
       PI(x) = P(f(x) < f_best)
       For a Gaussian posterior N(mu, sigma^2):
       PI = Phi((f_best - mu) / sigma)
       where Phi is the standard normal CDF.
       
    2. Cost Awareness:
       The cost of evaluating a point is given by `cost` (1 + mean(x_scaled)).
       We want to maximize the "value" per unit cost.
       Score = PI / cost
       
       Why PI over EI?
       - EI can be dominated by regions with high sigma but low probability of actually being better than f_best.
       - PI focuses on the probability mass of improvement, which is often more effective in multimodal functions like Rastrigin.
       - Dividing by cost penalizes expensive evaluations, encouraging the search towards cheaper regions (lower x values) if the improvement probability is comparable.
       
    3. Safety:
       - Ensure sigma is positive (clip to 1e-8).
       - Ensure finite outputs.
       - Check time limit.
    """
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
         # Return a neutral score to avoid hanging or invalid computation
         return np.zeros_like(mu)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero or numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute d = (f_best - mu) / sigma
    d = (f_best - mu) / sigma
    
    # Compute PI = Phi(d)
    PI = norm.cdf(d)
    
    # Ensure PI is within [0, 1] (numerical safety)
    PI = np.clip(PI, 0.0, 1.0)
    
    # Cost-aware score
    # We want to maximize this score.
    # Dividing by cost penalizes expensive points.
    cost_safe = np.maximum(cost, 1e-9)
    scores = PI / cost_safe
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
