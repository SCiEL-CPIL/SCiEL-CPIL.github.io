"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for the session
_SESSION_START = time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Adaptive UCB and PI exploration.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed function value so far.
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate, shape (2000,).
        
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Safety deadline guard: if we've used > 295s of the 330s limit,
    # return a simple fallback immediately.
    elapsed = time.time() - _SESSION_START
    if elapsed > 295:
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -10, 10)
        ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
        score = ei / cost_safe
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure sigma and cost are positive to avoid division by zero or invalid norms
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate z-score for Probability of Improvement and EI
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z_clipped)
    
    # Determine optimization progress
    # Rastrigin 2D on [-5.12, 5.12] has a max value of roughly 2*10 + 2*10*cos... 
    # Actually max is 2*10 + 2*10 = 40? No, 10*x^2 + 10*cos(2*pi*x).
    # Max x^2 is ~26. Max 10*x^2 is 260. Max cos is 1.
    # However, the "inner 80%" and typical BO runs often start with f_best around 10-30.
    # We use a reference scale of 40 to normalize progress.
    # As f_best drops to 0, progress goes to 1.
    ref_scale = 40.0
    progress = np.clip(1.0 - f_best / ref_scale, 0.0, 1.0)
    
    # Adaptive Beta for UCB
    # Early (progress=0): High exploration (beta=3.0)
    # Late (progress=1): Low exploration (beta=1.0)
    beta = 1.0 + 2.0 * (1.0 - progress)
    
    # Adaptive Weight for PI exploration bonus
    # Early (progress=0): High weight (2.0) to encourage finding new basins
    # Late (progress=1): Low weight (0.0) to focus on polishing
    pi_weight = 2.0 * (1.0 - progress)
    
    # Base UCB Score: -mu + beta * sigma
    # We want to maximize this.
    ucb_term = -mu + beta * sigma_safe
    
    # PI Exploration Bonus:
    # We scale PI by sigma to bring it to the same units as mu/sigma.
    # This rewards points that are both uncertain (high sigma) and likely to be better (high PI).
    pi_bonus = pi_weight * pi * sigma_safe
    
    # Combined Score
    # We sum the UCB term and the PI bonus.
    # Then we divide by cost to make it cost-aware.
    # Higher score = better expected gain per unit cost.
    
    raw_score = ucb_term + pi_bonus
    score = raw_score / cost_safe
    
    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
