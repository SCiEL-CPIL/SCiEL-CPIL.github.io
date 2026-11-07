"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    
    This is a structural change from Expected Improvement (EI) to UCB.
    UCB balances exploitation (low mean) and exploration (high standard deviation)
    more explicitly, which can help escape local optima in multimodal functions
    like Styblinski-Tang.
    
    Args:
        mu: array of shape (2000,) with posterior means
        sigma: array of shape (2000,) with posterior standard deviations
        f_best: lowest observed function value
    
    Returns:
        array of shape (2000,) with acquisition scores (higher is better)
    """
    # Guard against negative or zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Scale normalization to make the acquisition function invariant to the
    # magnitude of the objective function values.
    # We use the magnitude of the best value found so far as the scale.
    # This ensures that the exploration term is comparable to the exploitation term
    # regardless of whether the function values are O(1) or O(100).
    scale = max(abs(f_best), 1.0)
    
    # UCB parameter beta.
    # Higher beta encourages more exploration.
    # 2.0 is a standard choice that provides a good balance for multimodal optimization.
    beta = 2.0
    
    # UCB score for minimization: -mu + beta * sigma
    # Normalized by scale
    score = -mu / scale + beta * (sigma_safe / scale)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
