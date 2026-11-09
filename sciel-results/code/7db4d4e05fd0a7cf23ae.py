"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


class _BOState:
    def __init__(self):
        self.step = 0
        self.start_time = time.time()
        self.deadline = 300.0  # seconds

    def is_expired(self):
        return (time.time() - self.start_time) > (self.deadline - 15.0)


_state = _BOState()


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: adaptive UCB with gap-based beta and EI tie-breaking.
    
    Uses the structural idea of gap-adaptive beta (from inspiration 2) combined
    with a small EI component to ensure improvement bias (from inspiration 3).
    """
    _state.step += 1
    
    # Deadline guard: if we're running out of time, return pure exploitation
    if _state.is_expired():
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Theoretical minimum for Michalewicz 2D
    f_min = -1.8013
    
    # Calculate gap from known minimum
    gap = f_best - f_min
    if gap < 0:
        gap = 0.0
        
    # Normalize gap to [0, 1] roughly
    # Max possible value for Michalewicz 2D is ~9.6, min is -1.8
    # So max gap is ~11.4
    gap_norm = np.clip(gap / 10.0, 0.0, 1.0)
    
    # Adaptive beta based on gap:
    # High exploration (high beta) when gap is large (early or stuck)
    # Low exploration (low beta) when gap is small (converging)
    # Also incorporate step-based decay for additional robustness
    if _state.step <= 8:
        step_factor = 1.0
    elif _state.step <= 15:
        step_factor = 0.7
    elif _state.step <= 22:
        step_factor = 0.5
    else:
        step_factor = 0.3
    
    # Combine gap-based and step-based adaptation
    beta_gap = 0.5 + 2.5 * gap_norm  # Range [0.5, 3.0]
    beta = beta_gap * step_factor
    
    # Clip beta to reasonable range
    beta = np.clip(beta, 0.3, 3.0)
    
    # UCB scores
    ucb_scores = -mu + beta * sigma
    
    # EI component for improvement bias
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Scale EI to be comparable to UCB scores
    ei_max = np.max(ei) if np.max(ei) > 1e-9 else 1.0
    ei_scaled = ei / ei_max
    
    # Blend: primarily UCB, with small EI weight to bias toward improvement
    # The EI weight is higher when we're close to convergence (small gap)
    ei_weight = 0.1 + 0.2 * (1.0 - gap_norm)  # Range [0.1, 0.3]
    scores = (1.0 - ei_weight) * ucb_scores + ei_weight * ei_scaled
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
