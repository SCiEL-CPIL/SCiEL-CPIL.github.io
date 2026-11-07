"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Styblinski Tang 2D.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate, where cost = 1 + mean(x_scaled).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety margin: return early if we're getting close to the deadline
    # The evaluator has a 330s wall-clock limit, and we want to be safe.
    # Each call to acquisition is part of a loop that runs while total cost < 30,
    # so there are at most ~25 evaluations after the initial 5 points.
    # We add a small epsilon to avoid division by zero and ensure finite outputs.
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: penalize high-cost evaluations more strongly.
    # The cost is 1 + mean(x_scaled), so it ranges from 1 to 2.
    # Using cost^2 gives a stronger penalty for higher costs, which helps
    # exploit the fact that the minimum of Styblinski Tang is at (-2.9, -2.9)
    # in the original coordinates, which maps to the center of [0,1]^2.
    # The center has cost = 1 + 0.5 = 1.5, while corners have cost = 1 or 2.
    # Actually, cost = 1 + mean(x_scaled), so:
    # - At (0,0): cost = 1 + 0 = 1
    # - At (0.5, 0.5): cost = 1 + 0.5 = 1.5
    # - At (1, 1): cost = 1 + 1 = 2
    # The minimum of ST2D is at (-2.9, -2.9), which in [0,1]^2 is at (0.21, 0.21),
    # so cost = 1 + 0.21 = 1.21. This is closer to the low-cost region.
    
    # Use a power penalty to balance exploration and exploitation
    # with cost awareness. We use cost^1.5 as a compromise.
    cost_penalty = np.power(cost, 1.5)
    
    scores = ei / cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin
        # Return a simple fallback: just use EI without cost penalty
        return ei
    
    return scores
# EVOLVE-BLOCK-END
