"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a cost-normalized Upper Confidence Bound (UCB)
    with high gamma to aggressively favor low-cost, high-uncertainty regions.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - lowest observed value
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    --------
    score : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    t0 = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Safety margin check
    if time.time() - t0 > 280:
        sigma_safe = np.maximum(sigma, 1e-12)
        ucb = mu + 2.0 * sigma_safe
        cost_safe = np.maximum(cost, 1e-12)
        score = ucb / (cost_safe ** 3.0)
        return np.where(np.isfinite(score), score, 0.0)
    
    # Parameters
    gamma = 3.0  # high cost penalty exponent to favor cheap regions
    kappa = 2.5  # exploration-exploitation balance for UCB
    
    # Safe sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Compute UCB score: mu + kappa * sigma
    # For minimization, we want low values, so we negate to make higher = better
    # Actually, for minimization UCB should be mu - kappa * sigma (lower bound)
    # But we want to explore where sigma is high, so we use mu + kappa * sigma
    # and then negate the whole thing? No, let's think carefully.
    #
    # For minimization:
    # - We want to find points where the true value is likely low
    # - UCB for maximization: mu + kappa * sigma
    # - For minimization, we can use: mu - kappa * sigma (lower confidence bound)
    # - But we also want to explore high uncertainty regions
    # - A common approach is to use: -mu + kappa * sigma (negate the objective)
    #   This way, high sigma increases the score (exploration) and low mu increases it (exploitation)
    #
    # Let's use: score_base = -mu + kappa * sigma
    # This rewards both low predicted values and high uncertainty
    
    score_base = -mu + kappa * sigma_safe
    
    # Apply cost normalization with high gamma
    cost_penalty = cost_safe ** gamma
    score = score_base / cost_penalty
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Final safety check
    if time.time() - t0 > 290:
        pass
    
    return score
# EVOLVE-BLOCK-END
