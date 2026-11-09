import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0  # Safety margin before 330s limit


def _check_deadline():
    """Return True if we are approaching the deadline."""
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a combination of Expected Improvement (EI) and UCB with adaptive
    exploration/exploitation balance.
    
    Args:
        mu: Array of mean predictions, shape (N,)
        sigma: Array of standard deviation predictions, shape (N,)
        f_best: The best (lowest) observed function value so far.
        
    Returns:
        Array of scores, shape (N,). Higher scores indicate more promising candidates.
    """
    # Check deadline - if we're close, just return a simple UCB
    if _check_deadline():
        return -mu + sigma
    
    # Ensure arrays are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases: if sigma is all zeros or very small
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement (EI) for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For Gaussian: EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB for minimization: score = -mu + beta * sigma
    # Adaptive beta based on how close f_best is to the known minimum (0)
    # When f_best is large (far from optimum), explore more
    # When f_best is small (near optimum), exploit more
    
    # f_best for Weierstrass 1D on [-0.5, 0.5] ranges from 0 (minimum) to ~1+
    # Use a smooth transition
    f_best_clamped = np.clip(f_best, 0.0, 1.5)
    
    # Beta decreases as f_best approaches 0
    # beta ranges from ~4 (when f_best=1.5) to ~1 (when f_best=0)
    beta = 1.0 + 3.0 * (f_best_clamped / 1.5)
    
    ucb = -mu + beta * sigma
    
    # Combine EI and UCB with adaptive weighting
    # When EI is dominant (f_best is close to optimum), weight EI more
    # When UCB is dominant (early exploration), weight UCB more
    
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_max = np.max(ucb) if np.max(ucb) > 0 else 1.0
    
    ei_norm = ei / ei_max
    ucb_norm = ucb / ucb_max
    
    # Adaptive weight: more EI when f_best is small, more UCB when f_best is large
    # weight_ei ranges from 0.3 (f_best=1.5) to 0.8 (f_best=0)
    weight_ei = 0.3 + 0.5 * (1.0 - f_best_clamped / 1.5)
    weight_ucb = 1.0 - weight_ei
    
    scores = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Add a small bonus for points with very low predicted mean (exploitation)
    # This helps converge to the minimum
    mu_min = np.min(mu)
    mu_range = np.max(mu) - mu_min
    if mu_range > 1e-10:
        mu_bonus = (1.0 - (mu - mu_min) / mu_range) * 0.1
    else:
        mu_bonus = np.zeros_like(mu)
    
    scores = scores + mu_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
