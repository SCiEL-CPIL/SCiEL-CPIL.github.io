"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: Expected Improvement with a UCB exploration bonus,
    plus a small random perturbation to break ties and encourage diversity.
    
    Parameters:
    -----------
    mu : ndarray, shape (n_candidates,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (n_candidates,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        Acquisition scores (higher = better).
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB exploration bonus: use a moderate exploration coefficient
    # The coefficient balances exploitation (EI) and exploration (uncertainty)
    xi = 0.01  # small exploitation bias in EI
    ucb = mu + 1.5 * sigma_safe  # UCB with beta=1.5
    
    # Combine: primarily EI, with a small UCB component to encourage exploration
    # Weight them: 0.7 * EI + 0.3 * UCB (normalized)
    # First, normalize both to have similar scales
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    ei_range = ei_max - ei_min + 1e-9
    ucb_range = ucb_max - ucb_min + 1e-9
    
    ei_norm = (ei - ei_min) / ei_range
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Combine with weights
    scores = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Add small random noise to break ties and encourage exploration
    scores += 0.01 * np.random.randn(len(scores))
    
    return scores
# EVOLVE-BLOCK-END
