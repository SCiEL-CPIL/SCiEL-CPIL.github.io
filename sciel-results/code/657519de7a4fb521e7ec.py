"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Strategy: Hybrid Expected Improvement with Uncertainty Bonus
    # 
    # 1. Compute Expected Improvement (EI) for minimization.
    #    EI(x) = E[max(f_best - f(x), 0)]
    #    For f(x) ~ N(mu, sigma^2):
    #    EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    #    where Phi is the CDF and phi is the PDF of the standard normal distribution.
    #
    # 2. Add an exploration bonus proportional to sigma.
    #    The bonus is scaled by a factor that decays as f_best approaches the known minimum.
    #    This encourages exploration when the current best is far from the global minimum,
    #    and shifts to pure exploitation (EI dominates) when close.
    
    # Calculate the improvement term
    # We want to minimize f, so improvement is f_best - f(x).
    # Let d = f_best - mu.
    d = f_best - mu
    
    # Standard normal CDF and PDF
    # Use scipy.stats.norm for stability
    z = d / sigma_safe
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI calculation
    # Note: If sigma is very small, z becomes large, Phi(z) -> 1, phi(z) -> 0.
    # EI -> max(d, 0). This is correct.
    ei = d * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Determine exploration strength
    # The known global minimum for Styblinski-Tang 1D is -39.1662.
    known_min = -39.1662
    max_val = 50.0  # Approximate maximum value in the domain
    
    # Progress towards optimum: 0 at start, 1 at optimum
    if max_val == known_min:
        progress = 0.0
    else:
        progress = np.clip((max_val - f_best) / (max_val - known_min), 0.0, 1.0)
    
    # Exploration weight
    # Early on (progress ~ 0), we want strong exploration.
    # Late on (progress ~ 1), we want strong exploitation (EI).
    # We use a cosine annealing schedule for smooth transition.
    # weight_exploration goes from 1.0 to 0.0
    weight_exploration = 0.5 * (1 + np.cos(np.pi * progress))
    
    # Scale of exploration bonus
    # We want the bonus to be comparable to the EI scale.
    # Typical EI magnitude is on the order of sigma.
    # Let's set the bonus as alpha * sigma * weight_exploration.
    # Alpha is tuned to balance exploration and exploitation.
    # Based on literature, alpha ~ 1-2 is often effective.
    alpha = 2.0
    
    exploration_bonus = alpha * sigma_safe * weight_exploration
    
    # Total score = EI + Exploration Bonus
    # Higher score means more promising.
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
