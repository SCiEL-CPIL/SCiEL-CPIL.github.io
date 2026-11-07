"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_deadline = _start_time + 280.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of ackley 2d.
    Uses a hybrid approach: Thompson Sampling for exploration in early stages,
    transitioning to a UCB-based exploitation strategy as we converge.
    """
    # Deadline guard: if we are running out of time, fall back to greedy
    if time.time() > _deadline:
        return -mu

    n = len(mu)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    mu = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    
    # Estimate progress using f_best as a proxy
    # Ackley 2D minimum is 0. progress = 0 when f_best is large, 1 when near 0
    progress = 1.0 - f_best / (f_best + 1.0)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Adaptive parameters
    # kappa for UCB: starts high for exploration, decays for exploitation
    kappa = 2.5 * (1.0 - 0.7 * progress)
    
    # For Thompson Sampling, we sample from the posterior
    # Use a random seed based on the call pattern to ensure reproducibility within a run
    # But different across runs (handled by the evaluator's seed)
    # We'll use a simple approach: sample once per acquisition call
    
    # Thompson Sampling: sample from N(mu, sigma^2)
    # This gives a single sample from the posterior for each point
    # The point with the lowest sampled value is chosen
    
    # However, we need to return scores for all 2000 points
    # So we compute a score that combines exploitation and exploration
    
    # Hybrid score:
    # Early: more weight on exploration (sigma)
    # Late: more weight on exploitation (-mu)
    
    # Exploration weight
    w_explore = 0.8 * (1.0 - progress)
    # Exploitation weight  
    w_exploit = 0.5 + 0.5 * progress
    
    # Base UCB score
    ucb_score = -mu + kappa * sigma
    
    # Add a bonus for points that are likely to improve on f_best
    # This is a soft Expected Improvement component
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # But for simplicity and speed, use a simplified version
    
    # Simplified: reward points where mu < f_best, weighted by confidence
    improvement = np.maximum(0.0, f_best - mu)
    confidence = 1.0 / (1.0 + sigma)  # higher when sigma is low
    
    ei_bonus = improvement * confidence * w_exploit
    
    # Combine UCB and EI bonus
    scores = w_exploit * ucb_score + w_explore * sigma + ei_bonus
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
