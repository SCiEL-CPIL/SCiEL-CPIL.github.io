"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Rastrigin 2D.
    
    Uses a hybrid approach combining:
    1. Expected Improvement (EI) normalized by cost for exploitation
    2. Upper Confidence Bound (UCB) for exploration
    3. Adaptive weighting based on how far f_best is from the global minimum
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Check deadline - if we're running out of time, return a simple score
    if time.time() > _DEADLINE - 5.0:
        # Simple fallback: negative mu / cost
        cost_safe = np.maximum(cost, 1e-10)
        scores = -mu / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    n = len(mu)
    
    # Ensure inputs are safe
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x))/sigma(x)) + sigma(x) * phi((f_best - mu(x))/sigma(x))
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Compute Phi(z) and phi(z)
    # Using scipy would be ideal, but we can use numpy's approximation
    # For standard normal CDF, we use the error function
    from math import sqrt, erf
    
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    phi_z = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    
    # phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
    pdf_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * phi_z + sigma_safe * pdf_z
    
    # Cost-normalized EI
    ei_cost = ei / cost_safe
    
    # Compute UCB for exploration
    # UCB(x) = -mu(x) + beta * sigma(x)
    # beta is adaptive based on how far f_best is from 0
    # When f_best is large (far from minimum), we explore more
    # When f_best is small (close to minimum), we exploit more
    
    # Adaptive beta: starts high when f_best is large, decreases as f_best approaches 0
    if f_best > 50.0:
        beta = 4.0
    elif f_best > 10.0:
        beta = 3.0
    elif f_best > 2.0:
        beta = 2.0
    elif f_best > 0.5:
        beta = 1.5
    else:
        beta = 1.0
    
    ucb = -mu + beta * sigma_safe
    ucb_cost = ucb / cost_safe
    
    # Hybrid score: weighted combination of EI and UCB
    # Weight shifts from exploration to exploitation as f_best improves
    if f_best > 20.0:
        w_ei = 0.4
        w_ucb = 0.6
    elif f_best > 5.0:
        w_ei = 0.6
        w_ucb = 0.4
    elif f_best > 1.0:
        w_ei = 0.8
        w_ucb = 0.2
    else:
        w_ei = 0.9
        w_ucb = 0.1
    
    scores = w_ei * ei_cost + w_ucb * ucb_cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
