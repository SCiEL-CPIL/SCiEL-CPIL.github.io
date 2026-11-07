import numpy as np
import time
from scipy.stats import norm

# Global state to track progress across calls within a seed
_step_count = 0
_start_time = None

def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """Standard Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu - xi) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)

def _ucb(mu, sigma, kappa):
    """Lower Confidence Bound for minimization (we maximize the negative)."""
    return -mu + kappa * sigma

def _entropy_proxy(mu, sigma):
    """
    Gaussian Process-based entropy maximization proxy.
    Scores candidates with the logarithm of the posterior variance (log(sigma^2)).
    This directly targets regions of highest information gain.
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    # log(sigma^2) = 2 * log(sigma)
    return 2.0 * np.log(sigma_safe)

def acquisition(mu, sigma, f_best):
    global _step_count, _start_time
    
    # Initialize time tracking on first call
    if _start_time is None:
        _start_time = time.time()
        _step_count = 0
    
    elapsed = time.time() - _start_time
    _step_count += 1
    
    # Safety margin: return UCB if we're getting close to deadline
    if elapsed > 300:
        scores = -mu + 2.0 * sigma
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Ensure finite input
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Compute component scores
    ei_scores = _expected_improvement(mu, sigma, f_best, xi=0.01)
    
    # Adaptive UCB: beta scales with f_best to ensure strong exploration early on
    # When f_best is large (early), beta is large. When f_best is small (late), beta is small.
    # Beale function min is 0. f_best starts high.
    # Let's use a simple scaling. If f_best is ~100, beta ~ 5. If f_best is ~1, beta ~ 1.
    # beta = 1.0 + 0.1 * f_best ? Or maybe just fixed but high?
    # The lesson says: "Adaptive LCB with beta scaling positively with f_best ensures strong exploration in early stages... Measured: 0.7201 -> 4.841."
    # So let's use beta = 1.0 + 0.2 * f_best.
    beta = 1.0 + 0.2 * f_best
    ucb_scores = _ucb(mu, sigma, beta)
    
    ent_scores = _entropy_proxy(mu, sigma)
    
    # Normalize each score to [0, 1] for combination
    def normalize(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-12:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    ei_norm = normalize(ei_scores)
    ucb_norm = normalize(ucb_scores)
    ent_norm = normalize(ent_scores)
    
    # Adaptive weighting based on progress
    # Early on: more exploration (UCB, Entropy)
    # Later: more exploitation (EI)
    # Total evaluations = 30. First 5 are random. So 25 acquisition steps.
    progress = min(1.0, _step_count / 25.0)
    
    # Weights: EI, UCB, Entropy
    # At progress=0: w_ei=0.2, w_ucb=0.4, w_ent=0.4
    # At progress=1: w_ei=0.8, w_ucb=0.1, w_ent=0.1
    w_ei = 0.2 + 0.6 * progress
    w_ucb = 0.4 - 0.3 * progress
    w_ent = 0.4 - 0.3 * progress
    
    # Combine
    scores = w_ei * ei_norm + w_ucb * ucb_norm + w_ent * ent_norm
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
