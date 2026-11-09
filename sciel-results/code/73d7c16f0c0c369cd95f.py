import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we are taking too long, return a simple result.
    # The computation is very fast (numpy vectorized), but we add a safety margin.
    # 330s wall-clock limit. We check if we are close to the limit.
    # However, since this function is called many times, we just ensure it's fast.
    # The main timeout risk is the GP fitting which is done externally.
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    cost = np.asarray(cost, dtype=float).ravel()
    
    # Safety margins
    sigma = np.clip(sigma, 1e-12, None)
    cost = np.clip(cost, 1e-12, None)
    
    # Expected Improvement (minimization)
    delta = f_best - mu
    z = delta / sigma
    
    # Numerically stable EI calculation
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = delta * cdf_z + sigma * pdf_z
    
    # Ensure ei is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: combine EI/cost with a small penalty and exploration bonus
    # The EI/cost term rewards efficient improvement per unit cost
    # The penalty term slightly discourages expensive regions when EI is comparable
    # The sigma/cost term encourages exploration in high-uncertainty, cheap regions
    # 
    # Current best: score = ei / cost - 0.05 * cost + 0.05 * (sigma / cost)
    # 
    # Let's try to refine the balance. The exploration bonus helps escape local optima.
    # Perhaps we can use a softer cost penalty or a different exploration term.
    # 
    # Idea: Use a log-sum-exp style soft-min for the cost penalty?
    # Or simply tune the coefficients.
    # 
    # Let's try a slightly stronger exploration bonus and a weaker cost penalty.
    # score = ei / cost + alpha * (sigma / cost) - beta * cost
    # 
    # Current: alpha=0.05, beta=0.05
    # Let's try alpha=0.1, beta=0.02
    # Or maybe alpha=0.08, beta=0.03
    
    # Also, consider that the cost is in [1, 2].
    # ei / cost is in [0, something].
    # sigma / cost is in [0, something].
    
    # Let's try a different combination:
    # score = (ei + 0.1 * sigma) / cost - 0.02 * cost
    
    # This groups the exploration bonus with the improvement in the numerator,
    # making it also cost-aware.
    
    score = (ei + 0.1 * sigma) / cost - 0.02 * cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
