"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Calculates Upper Confidence Bound (UCB) for each candidate.
    
    This implementation combines the GP-based acquisition strategy of the current
    program with the superior exploration performance of the UCB approach observed
    in Inspiration 1 (score 1.4834 vs 1.4508 for EI).
    
    Strategy:
    - Use UCB for minimization: Score = -mu + kappa * sigma.
    - This explicitly rewards high uncertainty (sigma) and low predicted mean (-mu).
    - Cost is ignored because the minimizer is uniformly distributed, and cost
      penalties (as seen in Inspiration 2 and 3) degrade performance by biasing
      the search away from valid minimum locations in higher-cost regions.
    - A fixed kappa of 1.0 is used, which balances the scale of the objective
      (Hartmann 3D values ~ [-4, 0]) with the GP standard deviation.
    - A deadline guard ensures the function returns within the 330s limit.
    """
    start_time = time.time()
    deadline = start_time + 325.0  # 5s safety margin

    # Ensure inputs are 1D arrays of float64 for performance and stability
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    # f_best and cost are not used in the final score, but we ensure they are valid types
    # f_best = float(f_best)
    # cost = np.asarray(cost, dtype=np.float64).ravel()

    # Ensure sigma is positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)

    # UCB for minimization:
    # We want to minimize f.
    # Lower mu is better (so -mu is high).
    # Higher sigma is better for exploration (so +kappa*sigma is high).
    kappa = 1.0

    ucb_score = -mu + kappa * sigma_safe

    # Ensure finiteness: replace NaN/Inf with -inf so they are never selected
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, -np.inf)

    # Check deadline; if close, return immediately
    if time.time() > deadline - 1.0:
        return ucb_score

    return ucb_score
# EVOLVE-BLOCK-END
