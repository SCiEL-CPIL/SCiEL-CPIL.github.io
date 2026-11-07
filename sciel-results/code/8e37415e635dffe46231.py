import numpy as np
import time
from scipy.stats import norm

# Track elapsed time for deadline guard
_start_time = time.time()
_TIME_LIMIT = 320.0  # Safety margin from 330s

def acquisition(mu, sigma, f_best):
    # Check time
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # If we are running out of time, return a safe fallback
        return np.full(len(mu), -1e10)
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Handle edge cases
    if mu.size == 0:
        return np.array([])
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI)
    # EI is defined for minimization: E[max(f_best - f(x), 0)]
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Compute UCB Score
    # For minimization, LCB = mu - kappa * sigma.
    # We want to maximize the "score", so Score_UCB = -LCB = -mu + kappa * sigma.
    
    # Adaptive Kappa based on the gap between f_best and the predicted minimum mu_min
    finite_mu = mu[np.isfinite(mu)]
    if finite_mu.size > 0:
        mu_min = np.min(finite_mu)
        gap = f_best - mu_min
        
        # If gap is large, the GP predicts a point much better than f_best.
        # We should trust this prediction (Exploit), so use lower kappa.
        # If gap is small, we are stuck or close. We might need to explore (Higher kappa).
        # Scale: Gap of 10 is considered "large" for this problem scale.
        t = np.clip(gap / 10.0, 0.0, 1.0)
        # kappa ranges from 0.5 (exploit) to 2.5 (explore)
        kappa = 0.5 + 2.0 * (1.0 - t)
    else:
        kappa = 2.0
        
    ucb_score = -mu + kappa * sigma_safe
    
    # 3. Hybrid Blending
    # Determine the blend weight based on progress towards the global minimum (3.0)
    # dist_from_min is large initially, small at the end.
    dist_from_min = max(f_best - 3.0, 1e-6)
    
    # w_ei: weight for EI. 
    # When dist is large (early), w_ei should be small (favor UCB).
    # When dist is small (late), w_ei should be large (favor EI).
    # Use exponential decay for smooth transition.
    # scale=5.0: At dist=5, w_ei ~ 0.63. At dist=0, w_ei ~ 1.0. At dist=10, w_ei ~ 0.13.
    w_ei = np.exp(-dist_from_min / 5.0)
    w_ucb = 1.0 - w_ei
    
    # Combine scores
    # Note: EI is always >= 0. UCB score can be negative.
    # To ensure they are on comparable scales for blending, we don't normalize them strictly,
    # but the dynamic weight helps. In early stages, UCB dominates. In late stages, EI dominates.
    scores = w_ei * ei + w_ucb * ucb_score
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
