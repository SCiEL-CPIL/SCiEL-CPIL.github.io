import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses a hybrid approach combining Expected Improvement (EI) and UCB,
    with adaptive exploration. For minimization problems.
    
    Args:
        mu: array of shape (N,) - posterior mean
        sigma: array of shape (N,) - posterior standard deviation
        f_best: float - best observed value so far
    
    Returns:
        scores: array of shape (N,) - acquisition scores (higher = more promising)
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 320:
        return -mu
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement for minimization
    # EI(x) = E[min(0, f(x) - f_best)] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute UCB with adaptive beta
    # Use a moderate beta that balances exploration and exploitation
    # Given 30 evaluations total, we want more exploration early
    beta = 1.5
    ucb = -mu + beta * sigma
    
    # Normalize both to [0, 1] range for combination
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-10
    ei_norm = (ei - ei_min) / ei_range
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min + 1e-10
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Blend: weight EI more heavily but allow UCB to contribute
    # This helps explore high-uncertainty regions while still exploiting
    weight_ei = 0.6
    weight_ucb = 0.4
    
    scores = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Handle any numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
