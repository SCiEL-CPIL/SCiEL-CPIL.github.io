# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = 330.0
_start_time = None


def _reset_timer():
    global _start_time
    _start_time = time.time()


def _get_elapsed():
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses a hybrid of UCB and EI with adaptive exploration.
    """
    # Safety margin: ensure we finish well before 330s
    if _get_elapsed() > _DEADLINE - 2.0:
        # Return a safe fallback: pure exploration (highest sigma)
        return sigma.copy()

    n = len(mu)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z for EI
    z = (f_best - mu) / sigma_safe
    
    # EI component
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB component with adaptive kappa
    # For Weierstrass (multimodal), we need strong exploration
    # kappa=2.5 was measured to improve from 1.0 to 2.155
    # Let's use a slightly different approach: combine EI and UCB
    
    # Adaptive kappa based on how much exploration we've done
    # Early on, more exploration; later, more exploitation
    # Since we don't know the step number, use the spread of sigma
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # If sigma is very small everywhere, we're in exploitation mode
    # If sigma is large, we're in exploration mode
    exploration_ratio = sigma_mean / (sigma_max + 1e-9)
    
    # Blend EI and UCB
    # UCB: mu - kappa * sigma (minimization, so lower mu + higher sigma is better)
    # For minimization, UCB score = -mu + kappa * sigma
    # But we want higher score = better, so:
    # UCB_score = -mu + kappa * sigma
    
    # Let's use a weighted combination
    # Normalize EI and UCB to comparable scales
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-9
    
    kappa = 2.5
    ucb_scores = -mu + kappa * sigma_safe
    ucb_mean = np.mean(ucb_scores)
    ucb_std = np.std(ucb_scores) + 1e-9
    
    # Normalize both
    ei_norm = (ei - ei_mean) / ei_std
    ucb_norm = (ucb_scores - ucb_mean) / ucb_std
    
    # Blend: 50% EI, 50% UCB
    # This balances exploitation and exploration
    scores = 0.5 * ei_norm + 0.5 * ucb_norm
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
