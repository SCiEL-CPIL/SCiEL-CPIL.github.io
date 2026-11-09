"""Cost-aware acquisition for Ackley 2D Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a cost-aware exploration bonus.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    start_time = time.time()
    deadline = 330.0 - 5.0  # 5 second safety margin
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.clip(sigma, 1e-9, None)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Clip EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization: use sqrt(cost) for gentler penalty
    cost_safe = np.maximum(cost, 1e-9)
    cost_factor = np.sqrt(cost_safe)
    
    # Base score: EI per unit sqrt(cost)
    scores = ei / cost_factor
    
    # Add an exploration bonus based on uncertainty, also cost-normalized
    # This helps in early iterations when f_best is poor
    exploration_bonus = sigma_safe / cost_factor
    
    # Weight the exploration bonus by how far we are from the minimum
    # Ackley 2D minimum is 0, so if f_best is close to 0, reduce exploration
    # Use a soft threshold: weight = 1 / (1 + f_best)
    # But f_best can be negative? No, Ackley minimum is 0, so f_best >= 0
    # Actually, Ackley 2D has minimum 0, so f_best >= 0
    # Use weight = exp(-f_best) to smoothly reduce exploration as we improve
    exploration_weight = np.exp(-max(f_best, 0.0))
    
    scores += exploration_weight * exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return whatever we have
        return scores
    
    return scores
# EVOLVE-BLOCK-END
