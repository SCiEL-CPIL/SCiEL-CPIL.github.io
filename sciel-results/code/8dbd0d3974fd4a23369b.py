"""Cost-aware acquisition for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Hybrid LCB + EI acquisition, cost-aware, with a time guard.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions.
    sigma : ndarray, shape (2000,)
        GP standard deviations.
    f_best : float
        Best observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    deadline = 300.0  # safety margin within 330s wall-clock limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    n = len(mu)
    
    # --- LCB component: directly rewards low mu with uncertainty bonus ---
    # Use a beta that balances exploration; 2.0 is a common choice
    beta = 2.0
    lcb = -mu - beta * sigma
    
    # --- EI component: expected improvement over f_best ---
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # EI is non-negative by construction, but clip for safety
    ei = np.maximum(ei, 0.0)
    
    # --- Cost normalization ---
    # Divide by cost to get per-unit-cost score
    lcb_cost = lcb / cost
    ei_cost = ei / cost
    
    # --- Hybrid: combine LCB and EI ---
    # LCB explores under-explored regions; EI exploits near f_best
    # Weight them equally initially; the LCB term dominates in low-sigma regions
    # where EI is small, encouraging exploration of unvisited areas.
    score = lcb_cost + ei_cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard: if we're close to deadline, return what we have
    elapsed = time.time() - t0
    if elapsed > deadline - 5.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
