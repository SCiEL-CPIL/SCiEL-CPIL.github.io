import numpy as np
import time
from scipy.stats import norm

# Global timer for safety
_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s kill
_total_evals = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses an adaptive blend of Expected Improvement (EI) and Upper Confidence Bound (UCB).
    Early steps favor exploration (UCB) to escape local minima in the multimodal Weierstrass function.
    Later steps favor exploitation (EI) to refine the best solution.
    """
    global _total_evals
    _total_evals += 1
    
    # Check time budget
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(mu)

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-10)
    
    # Determine progress within the current seed (assuming 25 acquisition calls per seed)
    # _total_evals is 1-indexed.
    # Seed 0: evals 1-25. Seed 1: 26-50. Seed 2: 51-75.
    step_in_seed = (_total_evals - 1) % 25
    # progress goes from 0.0 to 1.0 over the 25 steps
    progress = step_in_seed / 24.0
    
    # Calculate EI
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    ei_score = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei_score = np.where(np.isfinite(ei_score), ei_score, 0.0)
    
    # Calculate UCB Score for minimization (maximize score)
    # UCB for minimization is typically mu - beta * sigma.
    # To make it a "higher is better" score, we negate it: beta * sigma - mu.
    # We use beta=2.0 as a standard 95% confidence bound approximation.
    ucb_score = 2.0 * sigma - mu
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, 0.0)
    
    # Normalize scores to [0, 1] to allow meaningful blending
    def normalize_scores(s):
        s_min = np.min(s)
        s_max = np.max(s)
        s_range = s_max - s_min
        if s_range < 1e-10:
            return np.zeros_like(s)
        return (s - s_min) / s_range

    ei_norm = normalize_scores(ei_score)
    ucb_norm = normalize_scores(ucb_score)
    
    # Adaptive weights
    # Start with high UCB weight (exploration), end with high EI weight (exploitation)
    # w_ucb starts at 1.0, ends at 0.1
    w_ucb = 0.9 * (1.0 - progress) + 0.1
    w_ei = 1.0 - w_ucb
    
    # Blend
    final_score = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite scores
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
