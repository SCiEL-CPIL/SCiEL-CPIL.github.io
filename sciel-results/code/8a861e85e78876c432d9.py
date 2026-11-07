"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_deadline = _start_time + 285.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of ackley 2d.
    Uses Knowledge Gradient (KG) approximation: expected reduction in the
    global minimum given an evaluation at each candidate point.
    """
    # Deadline guard: if we are running out of time, fall back to greedy
    if time.time() > _deadline:
        return -mu

    n = len(mu)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    mu = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    
    # Knowledge Gradient: E[max(0, f_best - Y_i)] where Y_i ~ N(mu_i, sigma_i^2)
    # This is the expected improvement over the current best for each point.
    # KG approximates the value of information from evaluating at point i.
    
    # For a single point, the expected reduction in the minimum is:
    # E[max(0, f_best - Y)] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF.
    
    # Compute z = (f_best - mu) / sigma
    safe_sigma = np.where(sigma > 1e-10, sigma, 1e10)
    z = (f_best - mu) / safe_sigma
    
    # Compute Phi(z) and phi(z)
    from scipy.stats import norm
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    # This is always non-negative and captures both the probability of improvement
    # and the magnitude of potential improvement.
    expected_improvement = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure non-negative (numerical safety)
    expected_improvement = np.maximum(expected_improvement, 0.0)
    
    # KG score: higher expected improvement means higher priority
    # This naturally balances exploration (high sigma -> high expected improvement)
    # and exploitation (mu < f_best -> high expected improvement)
    
    scores = expected_improvement
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
