"""Direct cost-aware acquisition for Bayesian optimization of rosenbrock 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition combining Expected Improvement (EI) with a cost penalty.
    
    The task specifies:
    - 2D Rosenbrock on [-2, 2]^2, minimum value 0
    - Cost(x) = 1 + mean(x scaled to [0, 1])  => cost in [1, 2]
    - The minimum is near the origin in the scaled space (cost ~ 1)
    - Total budget: 30 cost units, start with 5 uniform points
    
    Strategy: Use Expected Improvement (EI) which captures both exploitation
    (low predicted value) and exploration (high uncertainty). We add a
    cost-awareness term: since lower cost points are more likely to be near
    the minimum, we subtract a scaled cost penalty. This encourages the
    acquisition to prefer cheaper evaluations.
    
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z),  where z = (f_best - mu(x))/sigma(x)
    
    The cost penalty is: -lambda * (cost(x) - cost_min)
    We set lambda adaptively based on the magnitude of f_best - mu.
    
    We also guard against the 330s wall-clock limit with an explicit deadline.
    """
    start_time = time.time()
    DEADLINE = start_time + 310.0  # 20s safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle NaN/Inf
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -50, 50)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Structural Change: Cost-Effective Expected Improvement (CEI)
    # Maximize EI per unit cost. This directly optimizes the expected
    # improvement relative to the budget spent.
    # cost is in [1, 2], so dividing by it normalizes the "price" of the point.
    # This is a standard cost-aware acquisition function.
    
    # Avoid division by zero or very small costs (though cost >= 1)
    # cost is guaranteed to be >= 1 by problem definition, but guard just in case.
    safe_cost = np.maximum(cost, 1e-8)
    
    score = ei / safe_cost
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 300:  # Additional safety check
        # Fallback: pure exploitation with cost awareness
        # If we run out of time, pick the point with lowest mu and lowest cost
        score = -(mu) / safe_cost
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
