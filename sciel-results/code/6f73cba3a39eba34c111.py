# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) plus an exploration bonus (sigma),
    normalized by the evaluation cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of GP predictions in objective units.
    sigma : array-like, shape (2000,)
        Standard deviation of GP predictions in objective units.
    f_best : float
        Best observed value so far (minimum).
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 250.0  # 300s limit, keep 50s margin
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Sanitize inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e10, neginf=-1e10)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e10, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e10, neginf=1.0)
    
    # Ensure positive sigma and cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Check time
    if time.time() > deadline - 10.0:
        # Fallback: simple EI/cost or just return zeros if extremely late
        # For N=2000, this calculation is <1ms, so this is unlikely.
        pass

    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in cdf/pdf
    z = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration bonus
    # We add a term proportional to sigma to encourage exploration.
    # Coefficient 1.0 balances exploitation (EI) and exploration (sigma).
    exploration_bonus = 1.0 * sigma
    
    # Combine
    # Score is (EI + Exploration) / Cost
    # This favors points with high expected improvement or high uncertainty,
    # penalized by the cost of evaluation.
    scores = (ei + exploration_bonus) / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Final safety check
    if not np.all(np.isfinite(scores)):
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
        
    return scores
# EVOLVE-BLOCK-END
