"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm, t as t_dist

# Track start time for deadline enforcement
_START_TIME = time.time()
_DEADLINE_SECONDS = 325.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses a hybrid of Student-t Process (STP) approximation for heavier-tailed uncertainty
    and adaptive UCB, with progress-based blending.
    
    The STP component provides heavier tails for better exploration in the sharp
    curvature regions of the Beale function, while the adaptive UCB provides
    a simpler, robust exploration signal. The blend is adjusted based on
    optimization progress (f_best magnitude).
    """
    # Check deadline - if we're close to timeout, return a simple fallback
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # --- Student-t Process approximation component ---
    # Use nu = 3 degrees of freedom for heavier tails
    nu = 3.0
    
    # Compute EI under t-distribution
    z = (f_best - mu) / sigma_safe
    
    # Standard t CDF and PDF at z
    cdf_t = t_dist.cdf(z, nu, loc=0, scale=1)
    pdf_t = t_dist.pdf(z, nu, loc=0, scale=1)
    
    # Correction factor for t-distribution EI
    correction = (nu + z**2) / (nu - 1)
    
    ei_t = (f_best - mu) * cdf_t + sigma_safe * pdf_t * correction
    ei_t = np.maximum(ei_t, 0.0)
    
    # --- Adaptive UCB component ---
    # Adaptive beta: higher when f_best is large (early), lower when f_best is small (late)
    f_best_clamped = max(f_best, 1e-12)
    # Log-scale decay: beta = a + b / (1 + f_best)
    a, b = 0.3, 2.7
    beta = a + b / (1.0 + f_best_clamped)
    
    # Use t-distribution quantile for exploration with heavier tails
    # At 95% confidence, t-distribution has wider tails than normal
    beta_t = t_dist.ppf(0.95, nu, loc=0, scale=1)
    
    # Blend the two beta values: use the adaptive one but scale by t-quantile ratio
    # t-quantile at 95% for nu=3 is about 2.356, normal is about 1.645
    # So t-distribution naturally suggests more exploration
    beta_effective = beta * (beta_t / norm.ppf(0.95))
    
    ucb_t = -mu + beta_effective * sigma_safe
    
    # --- Normalize components for fair blending ---
    def normalize(arr):
        std = np.std(arr)
        if std < 1e-12:
            return arr - np.mean(arr)
        return (arr - np.mean(arr)) / std
    
    ei_norm = normalize(ei_t)
    ucb_norm = normalize(ucb_t)
    
    # --- Progress-based blending ---
    # progress = 0 when f_best is large, 1 when f_best is near 0
    progress = 1.0 / (1.0 + f_best_clamped)
    
    # Weights: UCB dominates early, EI dominates late
    # With STP, we can afford slightly more exploration even late
    w_ucb = 0.5 * (1.0 - progress) + 0.25 * progress  # 0.5 early, 0.25 late
    w_ei = 0.5 * (1.0 - progress) + 0.75 * progress   # 0.5 early, 0.75 late
    
    score = w_ucb * ucb_norm + w_ei * ei_norm
    
    # Handle potential numerical issues
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
