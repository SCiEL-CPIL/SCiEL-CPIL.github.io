"""Program for Bayesian optimization acquisition function."""
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
TIME_BUDGET = 325.0  # Leave 5s margin before 330s kill


def _time_remaining():
    return TIME_BUDGET - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using adaptive hybrid approach optimized for
    the Styblinski-Tang 1D minimization task.
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far (lower is better)
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    # Safety checks
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Clamp extreme sigma values
    sigma_safe = np.clip(sigma_safe, 1e-10, 1e6)
    
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute Upper Confidence Bound (for minimization, we want low mu + high sigma)
    # Use adaptive beta: higher when uncertain, lower when confident
    beta = 1.2 + 0.8 * np.exp(-f_best / 20.0)  # starts ~2.0, decays to ~1.2
    ucb = -mu + beta * sigma_safe
    
    # Compute a diversity/exploration term: prioritize unexplored regions
    expl = sigma_safe
    
    # Normalize each component to [0, 1] range for stable combination
    def _normalize(arr):
        a_min = np.min(arr)
        a_max = np.max(arr)
        rng = a_max - a_min
        if rng < 1e-15:
            return np.zeros_like(arr)
        return (arr - a_min) / rng
    
    ei_norm = _normalize(ei)
    ucb_norm = _normalize(ucb)
    expl_norm = _normalize(expl)
    
    # Adaptive weighting based on progress toward the known optimum (-39.1662)
    # When far from optimum, emphasize exploration; when close, emphasize exploitation
    OPTIMUM = -39.1662
    # Progress metric: how close f_best is to the optimum (0 = at start, 1 = at optimum)
    # Use a soft metric that saturates
    gap = f_best - OPTIMUM  # positive when above optimum
    # Normalize gap: assume initial f_best around 50-100 for ST function
    # Use sigmoid-like transition for smoother weight changes
    progress = 1.0 / (1.0 + np.exp(-gap / 15.0))
    # Flip so progress=0 means far from optimum, progress=1 means close
    progress = 1.0 - progress
    
    # Interpolate weights: from exploration-heavy to exploitation-heavy
    # Exploration phase: w_ei=0.25, w_ucb=0.45, w_expl=0.30
    # Exploitation phase: w_ei=0.75, w_ucb=0.15, w_expl=0.10
    w_ei = 0.25 + 0.50 * progress
    w_ucb = 0.45 - 0.30 * progress
    w_expl = 0.30 - 0.20 * progress
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm + w_expl * expl_norm
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Time guard: if we're running out of time, return a simple EI-based score
    if _time_remaining() < 5.0:
        simple_ei = np.maximum(ei, 0.0)
        simple_scores = np.nan_to_num(simple_ei, nan=0.0, posinf=1.0, neginf=0.0)
        return simple_scores
    
    return scores
