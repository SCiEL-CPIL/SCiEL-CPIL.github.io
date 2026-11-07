"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Upper Confidence Bound (UCB) for minimization.
    
    Structural Change: Replaces the hybrid EI/LCB with a pure UCB strategy.
    UCB balances exploitation (low mu) and exploration (high sigma) by
    maximizing the score: -mu - k * sigma.
    
    A high exploration constant k (set to 5.0) is chosen to aggressively
    explore unvisited regions of the domain, which is critical for the
    multimodal Michalewicz function to avoid getting trapped in local minima.
    This differs structurally from EI, which focuses on the expected magnitude
    of improvement, by focusing on the worst-case bound (with high probability).
    """
    # Time guard
    if time.time() > _DEADLINE - 5.0:
        return -np.where(np.isfinite(mu), mu, 0.0)

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # UCB for minimization:
    # We want to minimize f(x).
    # The lower confidence bound is L(x) = mu(x) - k * sigma(x).
    # We want to find the point with the lowest L(x).
    # To convert to a maximization problem (higher score is better),
    # we maximize -L(x) = -mu(x) + k * sigma(x).
    
    # Wait, standard UCB for maximization is U(x) = mu(x) + k * sigma(x).
    # For minimization, we minimize the Lower Bound: L(x) = mu(x) - k * sigma(x).
    # So we want to pick x with min L(x).
    # Score = -L(x) = -mu(x) + k * sigma(x).
    
    # Let's verify.
    # If mu is low (good), -mu is high (good score).
    # If sigma is high (uncertain), k*sigma is high (good score).
    # This encourages exploring high uncertainty areas.
    
    k = 5.0
    score = -mu + k * sigma_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
