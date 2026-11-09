"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill


def _time_check():
    """Check if we're close to the deadline."""
    return (time.time() - _START_TIME) > (_DEADLINE - 5.0)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses a hybrid approach:
    - Early phase (f_best large): UCB-like exploration to map the landscape
    - Late phase (f_best small): Expected Improvement for exploitation
    
    The transition is smooth based on f_best magnitude.
    """
    # Check deadline
    if _time_check():
        # Return a simple fallback: UCB with beta=0.5
        sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
        mu_safe = np.asarray(mu, dtype=np.float64)
        return 0.5 * sigma_safe - mu_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    
    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = 1.0
    f_best = max(abs(f_best), 1e-10)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Compute UCB score
    # Use a beta that adapts to the current f_best
    # When f_best is large, we want more exploration
    # When f_best is small, we want exploitation
    beta = 0.5 * np.exp(-f_best / 0.5)
    ucb = mu + beta * sigma
    ucb = np.where(np.isfinite(ucb), ucb, mu)
    
    # Also compute a pure exploitation score: negative mean
    exploit_score = -mu
    exploit_score = np.where(np.isfinite(exploit_score), exploit_score, 0.0)
    
    # Blend strategy:
    # - Use EI as the primary acquisition
    # - Add a small UCB component for exploration
    # - The weight of exploration decreases as f_best approaches 0
    
    # Normalize EI to be comparable with other terms
    # EI is always non-negative, higher is better
    
    # Adaptive blend:
    # exploration_weight goes from ~0.3 (early) to ~0.05 (late)
    exploration_weight = 0.3 * np.exp(-f_best / 1.0)
    
    # UCB component (scaled)
    ucb_component = beta * sigma
    
    # Final score: EI + exploration bonus
    scores = ei + exploration_weight * ucb_component
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Add a tiny bit of noise-breaking for ties (deterministic, based on index)
    # This helps avoid always picking the same point when scores are very close
    n = len(scores)
    tie_breaker = np.linspace(0, 1e-6, n)
    scores = scores + tie_breaker
    
    return scores
# EVOLVE-BLOCK-END
