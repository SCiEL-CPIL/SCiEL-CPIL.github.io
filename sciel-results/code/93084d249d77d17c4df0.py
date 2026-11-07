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
    Combines Lower Confidence Bound (LCB) for minimization with a strong cost penalty.
    
    mu: array of shape (2000,) - GP mean in objective units
    sigma: array of shape (2000,) - GP std dev in objective units
    f_best: float - best observed value so far
    cost: array of shape (2000,) - evaluation cost for each point, 1 + mean(x)
    
    Returns: array of shape (2000,) with acquisition scores (higher = better)
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle NaNs and Infs
    mu = np.nan_to_num(mu, nan=f_best, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-12)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Deadline Guard
    if time.time() - start_time > 290:
        return np.zeros_like(mu)

    # Expected Improvement (EI) for Minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma
    
    # Use scipy.stats.norm for CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Clip negative values due to numerical precision
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # Using power 1.0 for cost as a standard cost-efficiency metric
    score = ei / cost
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
