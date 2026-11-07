# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Fallback: return UCB-like scores quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 1.5 * sigma_safe
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Structural change: Use Probability of Improvement (PoI) as the primary exploitation signal
    # PoI is bounded in [0,1] and robust to scale variations
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)  # P(f(x) < f_best)
    
    # Compute Expected Improvement for additional exploitation signal
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB for minimization: -mu + beta * sigma
    # Adaptive beta: higher when uncertainty is high, lower when converging
    mean_sigma = np.mean(sigma_safe)
    std_sigma = np.std(sigma_safe)
    
    # Convergence indicator: fraction of points with predicted mean below f_best
    frac_better = np.mean(mu < f_best)
    
    # Adaptive beta: start high for exploration, decrease as we converge
    if mean_sigma > 0:
        # Relative uncertainty
        rel_uncert = std_sigma / (mean_sigma + 1e-9)
        # Base beta from relative uncertainty, modulated by convergence
        base_beta = 0.8 + 0.6 * min(rel_uncert, 1.0)
        # Scale down as we find better points
        beta = base_beta * (1.0 - 0.3 * frac_better)
        beta = np.clip(beta, 0.3, 2.0)
    else:
        beta = 1.0
    
    ucb = -mu + beta * sigma_safe
    
    # Normalize EI and UCB to [0,1] for combining
    def normalize(arr):
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb)
    
    # Sigma-rank exploration bonus: encourages visiting high-uncertainty regions
    sigma_rank = np.argsort(np.argsort(sigma_safe)).astype(float) / (n - 1) if n > 1 else np.zeros(n)
    
    # Combine signals with adaptive weights
    # When we're confident (high frac_better), weight PoI and EI more
    # When uncertain, weight UCB and sigma_rank more
    w_poi = 0.35 + 0.25 * frac_better
    w_ei = 0.25 + 0.15 * frac_better
    w_ucb = 0.25 - 0.15 * frac_better
    w_rank = 0.15  # Fixed exploration bonus
    
    # Ensure weights sum to 1
    total_w = w_poi + w_ei + w_ucb + w_rank
    w_poi /= total_w
    w_ei /= total_w
    w_ucb /= total_w
    w_rank /= total_w
    
    score = w_poi * poi + w_ei * ei_norm + w_ucb * ucb_norm + w_rank * sigma_rank
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
