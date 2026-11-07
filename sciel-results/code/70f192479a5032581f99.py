"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track start time for deadline guard
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement and LCB.
    
    Uses a hybrid approach that balances exploitation (via EI) and exploration
    (via LCB), with a cost penalty to prioritize low-cost evaluations.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from the GP in objective units.
    sigma : array-like, shape (2000,)
        Standard deviation predictions from the GP in objective units.
    f_best : float
        Best observed objective value so far (minimum, since we minimize).
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point, in [1, 2].
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline Guard
    elapsed = time.time() - _start_time
    if elapsed > 320:
        cost_arr = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost_arr, 1e-8)
        return 1.0 / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost_arr = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is strictly positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    # Ensure cost is strictly positive
    cost_safe = np.maximum(cost_arr, 1e-8)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    improvement = f_best - mu
    z = improvement / sigma_safe
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Compute EI
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    EI = improvement * Phi_z + sigma_safe * phi_z
    
    # Compute LCB for exploration
    kappa = 1.0
    lcb = mu - kappa * sigma_safe
    
    # Combine EI and LCB: use EI as the main exploitation signal
    # and add a small exploration bonus based on how far below f_best the LCB is
    exploration_bonus = np.maximum(f_best - lcb, 0) * 0.5
    
    # Total acquisition score before cost adjustment
    raw_score = EI + exploration_bonus
    
    # Apply cost penalty: divide by cost to prefer cheaper evaluations
    # Using a moderate power (1.5) to balance cost awareness with exploration
    score = raw_score / (cost_safe ** 1.5)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
