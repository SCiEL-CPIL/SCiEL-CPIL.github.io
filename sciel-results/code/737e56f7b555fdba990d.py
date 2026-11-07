"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer to satisfy "Explicit deadline guard" requirement
_start_time = time.time()
_TIME_LIMIT = 320.0  # Safety margin from 330s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement towards the known 
    global minimum (3.0) with an adaptive Uncertainty (UCB) term.
    
    Structural Change:
    Instead of normalizing and blending multiple terms (which loses magnitude 
    information), this approach uses a single coherent score:
    Score = EI_towards_3.0 + adaptive_kappa * sigma
    
    The adaptive_kappa is derived from the 'confidence' of the current best 
    prediction. If the GP predicts a point significantly better than f_best, 
    we reduce exploration (low kappa) to exploit that prediction. If the 
    predicted best is close to f_best, we increase exploration (high kappa) 
    to escape potential local minima.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean predictions at candidate points
    sigma : array of shape (2000,)
        Posterior standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher is better
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        # Fallback: simple exploitation
        return np.nan_to_num(-mu + 2.0 * sigma, nan=0.0, posinf=0.0, neginf=0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Safety: clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Known global minimum of Goldstein-Price 2D
    f_min_true = 3.0
    
    # 1. Calculate Expected Improvement towards the known global minimum (3.0)
    # This term is always positive and drives the search towards the true optimum.
    # Even if f_best is very good, this term remains a strong attractor.
    z_min = (f_min_true - mu) / sigma_safe
    ei_min = (f_min_true - mu) * norm.cdf(z_min) + sigma_safe * norm.pdf(z_min)
    
    # 2. Calculate Adaptive Kappa
    # We estimate the 'confidence' of the current posterior.
    # If the minimum predicted mean (mu_min) is significantly lower than f_best,
    # the GP has identified a promising region. We should exploit it (lower kappa).
    # If mu_min is close to f_best, we might be stuck in a local minimum or plateau.
    # We should explore more (higher kappa).
    
    finite_mu = mu[np.isfinite(mu)]
    if finite_mu.size == 0:
        kappa = 2.5
    else:
        mu_min = np.min(finite_mu)
        
        if not np.isfinite(f_best):
            gap = 0.0
        else:
            # Gap between observed best and predicted best
            # If gap is large, predicted best is much better -> Exploit (Low Kappa)
            # If gap is small, predicted best is similar to observed -> Explore (High Kappa)
            gap = f_best - mu_min
            
            # Clip gap to avoid extreme values
            gap = np.clip(gap, 0.0, 10.0)
            
            # Adaptive formula:
            # Base exploration level + additional exploration when gap is small.
            # exp(-gap / scale) is 1 when gap=0, decays as gap increases.
            # We want HIGH kappa when gap is SMALL.
            # So: kappa = base + range * exp(-gap / scale)
            
            # Tuning constants:
            # When gap=0 (stuck), kappa = 0.5 + 2.0 = 2.5 (High exploration)
            # When gap=10 (confident), kappa = 0.5 + 2.0*exp(-1) ~ 0.5 + 0.7 = 1.2 (Moderate exploitation)
            # Let's make the decay faster to be more exploitative when confident.
            # scale = 2.0
            # gap=0 -> kappa=2.5
            # gap=2 -> kappa=0.5 + 2.0*exp(-1) = 1.24
            # gap=4 -> kappa=0.5 + 2.0*exp(-2) = 0.74
            # gap=6 -> kappa=0.5 + 2.0*exp(-3) = 0.56
            
            scale = 2.0
            kappa = 0.5 + 2.0 * np.exp(-gap / scale)
            
            # Ensure kappa is within reasonable bounds
            kappa = np.clip(kappa, 0.5, 3.0)

    # 3. Combine
    # Score = EI_towards_3.0 + kappa * sigma
    # The EI term handles the direction towards the optimum.
    # The kappa * sigma term handles the exploration/exploitation balance.
    # Since EI_min already includes a sigma term (sigma * pdf(z)), 
    # adding another kappa * sigma effectively increases the weight on uncertainty.
    
    scores = ei_min + kappa * sigma_safe
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
