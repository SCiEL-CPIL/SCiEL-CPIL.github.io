"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 325.0  # Safety margin before the 330s kill limit
_eval_count = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses a Pareto-based multi-objective approach combining normalized
    Expected Improvement and normalized predictive uncertainty with
    a dynamic weight that shifts from exploration to exploitation
    as the number of evaluations increases.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    global _eval_count
    
    # Check for timeout
    if time.time() - _start_time > _TIMEOUT:
        return np.zeros_like(mu)
    
    _eval_count += 1
    
    # Ensure sigma is non-negative and finite, avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # --- Objective 1: Expected Improvement (exploitation) ---
    z = (f_best - mu) / sigma
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # --- Objective 2: Predictive Uncertainty (exploration) ---
    # Use sigma directly as the uncertainty measure
    uncertainty = sigma
    
    # --- Normalize both objectives to [0, 1] range ---
    # For EI: normalize by max EI (or a robust scale)
    ei_max = np.max(ei)
    if ei_max > 1e-10:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # For uncertainty: normalize by max sigma
    sigma_max = np.max(uncertainty)
    if sigma_max > 1e-10:
        unc_norm = uncertainty / sigma_max
    else:
        unc_norm = np.zeros_like(uncertainty)
    
    # --- Dynamic weight: shift from exploration to exploitation ---
    # Early on (few evals), prioritize exploration (higher weight on uncertainty)
    # Later (more evals), prioritize exploitation (higher weight on EI)
    # Use a smooth transition based on evaluation count
    # Total budget is 30 evaluations, so normalize by that
    total_budget = 30
    progress = min(_eval_count / total_budget, 1.0)
    
    # Weight for exploitation increases with progress
    # At start: w_exploit ~ 0.3, w_explore ~ 0.7
    # At end:   w_exploit ~ 0.8, w_explore ~ 0.2
    w_exploit = 0.3 + 0.5 * progress
    w_explore = 1.0 - w_exploit
    
    # --- Combine objectives ---
    scores = w_exploit * ei_norm + w_explore * unc_norm
    
    # Clean up non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
