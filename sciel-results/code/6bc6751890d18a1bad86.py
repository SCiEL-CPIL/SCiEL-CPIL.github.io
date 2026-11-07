# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE_START = time.time()
_SAFETY_MARGIN = 25.0

def _time_left():
    return 300.0 - (time.time() - _DEADLINE_START) - _SAFETY_MARGIN

def _expected_improvement(mu, sigma, f_best):
    z = (f_best - mu) / (sigma + 1e-12)
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

def _optimistic_lcb(mu, sigma, beta):
    # For minimization, the optimistic lower bound is mu - beta * sigma.
    # We want higher score to be better, so we negate it: -mu + beta * sigma.
    return -mu + beta * sigma

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)

    n = mu.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    # Adaptive beta: higher uncertainty in the population suggests we are early,
    # so we use a higher beta to encourage exploration.
    sigma_mean = float(np.mean(sigma))
    # Map sigma_mean to beta range [1.5, 3.0]
    # If sigma_mean is large, beta is high.
    beta = 1.5 + 1.5 * (sigma_mean / (sigma_mean + 1.0))
    
    # EI component
    ei = _expected_improvement(mu, sigma, f_best)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Optimistic LCB component
    lcb = _optimistic_lcb(mu, sigma, beta)
    lcb = np.where(np.isfinite(lcb), lcb, 0.0)
    
    # Normalize both to [0, 1]
    def _normalize(a):
        a = np.asarray(a, dtype=float)
        lo = float(np.min(a))
        hi = float(np.max(a))
        if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)
        
    norm_ei = _normalize(ei)
    norm_lcb = _normalize(lcb)
    
    # Blend: 60% Exploration (LCB), 40% Exploitation (EI)
    # This favors the Optimistic LCB which was previously broken.
    explore_weight = 0.6
    exploit_weight = 0.4
    
    score = explore_weight * norm_lcb + exploit_weight * norm_ei
    
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
