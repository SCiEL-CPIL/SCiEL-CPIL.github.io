import numpy as np
import time
from scipy.stats import norm

# Hartmann 6D function
def _hartmann_6d(x):
    """Compute Hartmann 6D objective value (lower is better)."""
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])
    
    val = np.zeros(len(x))
    for i in range(len(x)):
        s = 0.0
        for j in range(4):
            s += alpha[j] * np.sum(A[j] * (x[i] - P[j]) ** 2 * np.cos(np.pi * A[j] * (x[i] - P[j])))
        val[i] = -np.exp(-s)
    return val

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Hartmann 6D.
    Uses UCB (Lower Confidence Bound for minimization) with linear cost penalty.
    
    mu: array of shape (2000,) - GP mean in objective units
    sigma: array of shape (2000,) - GP std dev in objective units
    f_best: float - best observed value so far
    cost: array of shape (2000,) - evaluation cost for each point, 1 + mean(x)
    
    Returns: array of shape (2000,) with acquisition scores (higher = better)
    """
    # Ensure finite inputs
    mu = np.nan_to_num(mu, nan=f_best, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-9)
    
    # UCB parameter
    # Beta controls the trade-off between exploitation (mu) and exploration (sigma)
    # For minimization, we use LCB = mu - beta * sigma
    # We want to maximize -LCB = beta * sigma - mu
    beta = 2.0
    
    # Calculate UCB score (higher is better for selection)
    # Score = (beta * sigma - mu) / cost
    # Dividing by cost penalizes expensive evaluations
    # Linear cost penalty as per directive
    score = (beta * sigma - mu) / cost
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
