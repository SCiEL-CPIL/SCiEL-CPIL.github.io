"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


class BOState:
    def __init__(self):
        self.step = 0
        self.f_best_history = []
        self.start_time = time.time()
        self.deadline = 300.0  # seconds
        self.rng = np.random.default_rng(42)

    def is_expired(self):
        return (time.time() - self.start_time) > (self.deadline - 10.0)


state = BOState()


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling for Michalewicz 2D.
    
    Uses Thompson Sampling: sample from the posterior GP at candidate points,
    then select the point with the lowest sampled value. This provides natural
    exploration-exploitation balance and helps escape local minima.
    
    Returns 2000 finite scores, higher meaning evaluate next.
    """
    state.step += 1
    
    # Deadline guard: if we're running out of time, return pure exploitation
    if state.is_expired():
        scores = -mu  # Pure exploitation: pick lowest mean
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Track f_best history
    state.f_best_history.append(f_best)
    if len(state.f_best_history) > 40:
        state.f_best_history = state.f_best_history[-40:]
    
    # Adaptive exploration weight based on step
    # Early: more exploration, Late: more exploitation
    if state.step <= 8:
        beta = 1.5
    elif state.step <= 15:
        beta = 1.0
    elif state.step <= 22:
        beta = 0.5
    else:
        beta = 0.2
    
    # Thompson Sampling: draw a sample from the posterior at each candidate
    # For minimization, we want to find points likely to have low values
    # Sample: y = mu + beta * sigma * noise
    noise = state.rng.standard_normal(len(mu))
    ts_scores = mu + beta * sigma * noise
    
    # Higher score = more likely to be evaluated next
    # We want to pick points with LOW predicted values, so negate
    scores = -ts_scores
    
    # Add a small EI component to bias toward improvement over f_best
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Scale EI to be comparable
    ei_max = np.max(ei) if np.max(ei) > 1e-9 else 1.0
    ei_scaled = ei / ei_max
    
    # Blend: primarily Thompson Sampling, with EI as a tiebreaker/bias
    # The EI component helps ensure we're still improving on f_best
    blend = 0.85  # Weight for Thompson Sampling
    scores = blend * scores + (1.0 - blend) * ei_scaled
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
