import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization (minimization).
    Uses a combination of Expected Improvement and UCB for robustness.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin from 300s limit

    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)

    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid overflow in exp
    z = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Compute UCB for minimization
    # We want to explore where sigma is high and mu is low
    # Score = mu - beta * sigma
    # Adaptive beta: start high, decrease as we gather more info
    # Since we don't know N directly, we can estimate exploration vs exploitation
    # A fixed beta around 2-4 works well for small N in 2D
    beta = 3.0
    ucb = mu - beta * sigma_safe

    # Combine EI and UCB
    # Normalize both to [0, 1] range roughly, then combine
    # Or simply take a weighted sum
    # EI is already in objective units, UCB is in objective units
    # Let's use a weighted sum with equal weight initially
    # But EI can be negative (if mu > f_best), UCB can be anything
    # Better: use a rank-based combination or just pick the better one
    
    # Actually, let's use a simpler, more robust approach:
    # Use EI as the primary score, but add a small exploration bonus
    # The exploration bonus is proportional to sigma
    
    # Let's try: score = EI + gamma * sigma
    # where gamma is small, like 0.1
    gamma = 0.1
    scores = ei + gamma * sigma_safe

    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    return scores
