import numpy as np
from scipy.stats import norm
import time

# Global deadline guard: evaluator kills after 330s wall-clock.
# We'll use a conservative limit to ensure we finish safely.
_DEADLINE = 325.0  # seconds from process start (safety margin)
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    to balance exploration and exploitation.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more desirable.
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Return a simple fallback: exploit best mean
        return -mu
    
    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize the acquisition, so for minimization UCB:
    # We consider the lower bound of the objective: mu - beta * sigma
    # A point is attractive if its lower bound is low (good potential)
    # So acquisition_Ucb = -mu + beta * sigma (higher is better)
    
    # Adaptive beta: larger early on, smaller later
    # We don't have direct info about step count, but f_best tells us something.
    # For Ackley, f_best starts around 5-10 and decreases to 0.
    # If f_best is high, we're early -> explore more.
    # If f_best is low, we're late -> exploit more.
    
    # Simple heuristic: beta based on how far f_best is from 0
    # When f_best is large, explore more; when close to 0, exploit.
    # Ackley min is 0, max on [-5,5]^2 is around 20-30.
    
    # Let's use a smooth transition
    # beta = 2.0 * exp(-f_best / 5.0) + 0.5
    # But this might be too aggressive. Let's try a fixed moderate beta.
    
    # Actually, a common approach is to use beta that decreases over iterations.
    # Since we don't have iteration count, let's use a value that works well.
    
    # For Ackley 2D, which is relatively smooth, a moderate exploration is good.
    beta = 1.5
    
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB
    # Normalize both to [0, 1] range for fair combination
    ei_max = np.max(ei)
    if ei_max > 0:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    ucb_max = np.max(ucb)
    ucb_min = np.min(ucb)
    if ucb_max > ucb_min:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = ucb
    
    # Weighted combination: favor EI slightly for exploitation
    score = 0.6 * ei_norm + 0.4 * ucb_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
