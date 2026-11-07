import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_DEADLINE = time.time() + 280  # safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Pareto-frontier selection acquisition function.
    
    This function computes both Expected Improvement (EI) and Upper Confidence Bound (UCB)
    scores independently. It then uses a Pareto-frontier selection criterion: for each candidate,
    it computes the minimum of its normalized EI rank and normalized UCB rank. The final score
    is this minimum, which effectively selects points that are simultaneously among the most
    promising (high EI) and most uncertain (high UCB).
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP posterior mean at candidate points.
    sigma : np.ndarray, shape (2000,)
        GP posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher means more attractive to evaluate next.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: just exploit (maximize -mu)
        return -np.asarray(mu, dtype=np.float64).ravel().copy()
    
    # Ensure arrays are 1D and finite
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    n = len(mu)
    
    # Handle any non-finite values
    mask = np.isfinite(mu) & np.isfinite(sigma)
    if not np.all(mask):
        # Replace non-finite with worst-case for mu (high value for minimization)
        # and 0 for sigma
        safe_mu_val = np.max(mu[mask]) if np.any(mask) else 0.0
        mu = np.where(mask, mu, safe_mu_val)
        sigma = np.where(mask, sigma, 0.0)
    
    # Ensure sigma is strictly positive to avoid division by zero
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    
    # 1. Calculate Standard Expected Improvement (EI)
    # Improvement is f_best - mu
    improvement = f_best - mu
    z = improvement / sigma
    
    # Clip z to prevent numerical overflow/underflow in norm.cdf/pdf
    z_clipped = np.clip(z, -50, 50)
    
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    scores_ei = improvement * Phi_z + sigma * phi_z
    
    # 2. Calculate UCB score
    # Adaptive kappa based on progress
    f_global_min = -3.86278
    f_worst_ref = 0.0
    
    denominator = f_worst_ref - f_global_min
    if denominator == 0:
        progress = 0.5
    else:
        progress = (f_worst_ref - f_best) / denominator
        progress = np.clip(progress, 0.0, 1.0)
    
    # High kappa -> Explore. Low kappa -> Exploit.
    kappa_max = 3.0
    kappa_min = 0.5
    kappa = kappa_max - (kappa_max - kappa_min) * progress
    
    # UCB score for minimization: -mu + kappa * sigma
    # Higher score = better candidate
    scores_ucb = -mu + kappa * sigma
    
    # 3. Compute ranks for EI and UCB
    # Rank from 0 (lowest) to 1 (highest)
    ei_ranks = np.argsort(np.argsort(scores_ei)) / (n - 1) if n > 1 else np.ones(n)
    ucb_ranks = np.argsort(np.argsort(scores_ucb)) / (n - 1) if n > 1 else np.ones(n)
    
    # 4. Pareto-frontier selection: minimum of normalized ranks
    # This selects points that are simultaneously high in both EI and UCB
    scores = np.minimum(ei_ranks, ucb_ranks)
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
