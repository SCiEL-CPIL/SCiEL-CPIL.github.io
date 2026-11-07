import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Deterministic cost-penalized Expected Improvement (EI) acquisition.
    
    Strategy: Use raw EI (Expected Improvement for minimization) and subtract
    a small linear cost penalty. This treats the acquisition as maximizing
    EI - lambda * cost, which deterministically guides the search toward
    cheaper regions without the stochastic noise of UCB or the aggressive
    bias of pure cost-agnostic EI.
    
    The EI component naturally balances exploration and exploitation as the
    GP posterior updates. The cost penalty is small (lambda=0.1) to avoid
    over-biasing toward cheap regions while still accounting for evaluation cost.
    """
    # Deadline guard: if we're close to the time limit, return a simple fallback
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback: simple greedy exploitation with cost penalty
        scores = -mu - 0.1 * cost
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for EI (minimization)
    # z = (f_best - mu) / sigma
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Clip z to prevent numerical issues in CDF/PDF
    z_clipped = np.clip(z, -30, 30)
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    ei = diff * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical artifacts might make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost penalty: subtract lambda * cost
    # lambda = 0.1 provides a mild penalty for expensive evaluations
    lambda_cost = 0.1
    scores = ei - lambda_cost * cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
