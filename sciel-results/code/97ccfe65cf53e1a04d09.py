import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 290.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a small
    UCB exploration bonus for minimization of the Goldstein-Price 2D function.
    
    EI provides a theoretically grounded balance between exploration and
    exploitation by directly maximizing the expected gain over f_best.
    A small UCB component helps escape local minima by boosting candidates
    in high-uncertainty regions where EI might be low but potential improvement
    is high.
    
    EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    where Z = (f_best - mu) / sigma
    
    The combined score is EI + kappa_ucb * sigma, where kappa_ucb is small
    to ensure EI remains the primary signal while allowing some exploration.
    """
    # Time guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        return np.zeros_like(mu)
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement for minimization
    Z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    Phi = norm.cdf(Z)
    phi = norm.pdf(Z)
    
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    EI = (f_best - mu) * Phi + sigma_safe * phi
    
    # Clip small negative values to zero (EI should be non-negative)
    EI = np.maximum(EI, 0.0)
    
    # Add a small UCB component for exploration
    # This helps in regions where EI is low but uncertainty is high
    # Use a small coefficient to not dominate the EI signal
    kappa_ucb = 0.3
    UCB_bonus = kappa_ucb * sigma_safe
    
    # Combine: primarily EI, with a small UCB boost
    scores = EI + UCB_bonus
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
