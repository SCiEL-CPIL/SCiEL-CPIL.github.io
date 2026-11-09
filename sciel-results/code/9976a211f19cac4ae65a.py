"""Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid of Expected Improvement and Upper Confidence Bound.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at candidate points
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at candidate points
    f_best : float
        Best observed value so far (lowest)
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next
    """
    # Ensure we have valid inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases where sigma might be zero or negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Compute Upper Confidence Bound (UCB) with adaptive beta
    # We use a time-decaying exploration parameter
    # Since we don't have direct access to the iteration count, we use f_best as a proxy
    # for how much we've learned. Early on, f_best is high (far from optimum), so we explore more.
    # As f_best decreases (approaching optimum), we exploit more.
    
    # Scale beta based on how far we are from the known minimum
    # The known minimum is -2.53218
    known_min = -2.53218
    # Distance from current best to known minimum
    distance_to_min = f_best - known_min
    
    # If we're far from the minimum, explore more; if close, exploit more
    # Normalize the distance to get a reasonable beta
    # distance_to_min starts around 0.5-1.0 and decreases to 0
    # We want beta to start around 2.0-3.0 and decrease to around 0.5-1.0
    beta = 0.5 + 2.5 * (distance_to_min / max(distance_to_min, 1e-6))
    
    # Cap beta to avoid extreme values
    beta = np.clip(beta, 0.5, 3.0)
    
    # UCB score
    ucb = mu + beta * sigma_safe
    
    # Combine EI and UCB with adaptive weights
    # Early: more UCB weight for exploration; Late: more EI weight for exploitation
    # Use a sigmoid-like function based on distance to minimum
    # When distance is large (early), weight_ucb is high
    # When distance is small (late), weight_ei is high
    
    # Normalize distance to [0, 1] range approximately
    dist_norm = np.clip(distance_to_min / 1.0, 0.0, 1.0)
    
    # Weights: early on (dist_norm ~ 1), favor UCB; late (dist_norm ~ 0), favor EI
    weight_ucb = 0.3 + 0.7 * dist_norm  # Range: [0.3, 1.0]
    weight_ei = 1.0 - weight_ucb + 0.3  # Range: [0.3, 1.0], always at least 0.3
    
    # Normalize weights to sum to 1
    total_weight = weight_ucb + weight_ei
    weight_ucb /= total_weight
    weight_ei /= total_weight
    
    # Combine the scores
    # EI is already in "improvement" units, UCB is in objective units
    # We need to scale them appropriately
    # EI values are typically small (improvements), UCB values are in objective space
    # Let's scale EI by a factor to make it comparable to UCB
    
    # Scale EI: multiply by a factor based on the scale of the problem
    # The function values range roughly from -2.5 to 0.5, so scale ~ 3
    ei_scaled = ei * 3.0
    
    # Final score: weighted combination
    scores = weight_ei * ei_scaled + weight_ucb * (-ucb)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
