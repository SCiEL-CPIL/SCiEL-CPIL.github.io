"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware hybrid acquisition function combining EI and UCB.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-scores for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB (Upper Confidence Bound) - exploration term
    # Beta adapts based on how close we are to the minimum
    # For Rastrigin, the minimum is 0, so as f_best approaches 0, we exploit more
    f_best_abs = abs(f_best)
    
    # Adaptive beta: start with more exploration, reduce as we improve
    # When f_best is large (far from minimum), explore more
    # When f_best is small (close to minimum), exploit more
    beta = 0.5 + 0.5 * np.exp(-f_best_abs / 5.0)
    
    ucb = mu + beta * sigma_safe
    
    # Hybrid: combine EI and UCB
    # EI rewards improvement, UCB rewards uncertainty
    # Normalize both to comparable scales
    ei_norm = ei / (np.max(ei) + 1e-9) if np.max(ei) > 0 else ei
    ucb_norm = -ucb / (np.max(np.abs(ucb)) + 1e-9)  # Negate because lower is better
    
    # Weighted combination
    weight_ei = 0.6
    weight_ucb = 0.4
    
    combined = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Cost-aware normalization: divide by cost to prefer cheap evaluations
    # Use a soft normalization to avoid extreme values
    cost_factor = 1.0 / (cost_safe + 0.1)
    
    scores = combined * cost_factor
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - 5.0:
        # Return best result found with safety margin
        return scores
    
    return scores
# EVOLVE-BLOCK-END
