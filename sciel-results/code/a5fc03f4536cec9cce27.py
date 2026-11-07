"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global timer for deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # 5s margin from 330s

def _check_deadline():
    return (time.time() - _start_time) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if _check_deadline():
        # Fallback: prioritize exploitation (lowest mu) with slight cost awareness
        # If we are running out of time, we want to be safe and deterministic.
        # Using -mu is standard exploitation. 
        # We add a small penalty for cost to avoid expensive points if possible,
        # but primarily we want the best known value.
        return -mu / (cost + 1e-9)

    # Thompson Sampling Strategy:
    # 1. Draw a single scalar random variable z ~ N(0, 1)
    # 2. Compute sampled values: theta = mu + z * sigma
    #    Note: Using a single z for all points ensures that the relative ordering
    #    of the samples is driven by the relative magnitudes of mu and sigma,
    #    effectively acting as a randomized UCB where the "randomness" is shared.
    #    This is a common approximation for TS in vectorized settings to avoid
    #    breaking the correlation structure or requiring full covariance sampling.
    #    However, true TS samples independent values for each point.
    #    Let's use independent samples for each point to be a proper Thompson Sample.
    #    theta_i ~ N(mu_i, sigma_i^2)
    
    # Generate 2000 independent samples
    # np.random.normal is fast enough for 2000 samples
    z = np.random.normal(0, 1, size=mu.shape)
    theta = mu + z * sigma
    
    # Score: We want to maximize the score.
    # Since we are minimizing the objective, lower theta is better.
    # So score = -theta.
    # Cost-aware: Divide by cost.
    # score = -theta / cost
    
    # Ensure cost is positive and handle division
    scores = -theta / (cost + 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
