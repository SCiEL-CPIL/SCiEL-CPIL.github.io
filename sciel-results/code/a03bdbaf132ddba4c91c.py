"""Rewritten program for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    with adaptive exploration parameter based on the optimization progress.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean predictions
    sigma : ndarray, shape (2000,)
        Posterior standard deviation predictions
    f_best : float
        Best observed value so far
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores, higher means evaluate next
    """
    # Time guard: if we're running out of time, fall back to pure exploitation
    elapsed = time.time() - _start_time
    if elapsed > 325:
        return -mu
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    
    # Ensure mu is finite
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu_safe) / sigma_safe
    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)
    
    # Phi(z) = CDF, phi(z) = PDF using error function
    Phi_z = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
    phi_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    # EI calculation for minimization
    ei = np.zeros_like(mu_safe)
    mask = mu_safe < f_best
    ei[mask] = (f_best - mu_safe[mask]) * Phi_z[mask] + sigma_safe[mask] * phi_z[mask]
    
    # Compute UCB for minimization
    # Adaptive kappa: start with more exploration, reduce as we find better solutions
    # Use a simple heuristic based on the spread of mu values
    mu_range = np.ptp(mu_safe)
    if mu_range > 1e-10:
        # Normalize f_best relative to the range
        # If f_best is near the minimum of mu, we're doing well, reduce exploration
        mu_min = np.min(mu_safe)
        progress = (f_best - mu_min) / (mu_range + 1e-10)
        progress = np.clip(progress, 0.0, 1.0)
        # kappa decreases from ~1.0 to ~0.3 as progress increases
        kappa = 1.0 - 0.7 * progress
    else:
        kappa = 0.5
    
    ucb = -mu_safe + kappa * sigma_safe
    
    # Combine EI and UCB
    # Normalize both to comparable scales
    ei_max = np.max(ei)
    if ei_max > 1e-10:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-10:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Weighted combination: favor EI more as optimization progresses
    # Early on, UCB helps explore; later, EI drives exploitation
    if progress < 0.3:
        w_ei = 0.5
        w_ucb = 0.5
    elif progress < 0.6:
        w_ei = 0.7
        w_ucb = 0.3
    else:
        w_ei = 0.85
        w_ucb = 0.15
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite scores
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
