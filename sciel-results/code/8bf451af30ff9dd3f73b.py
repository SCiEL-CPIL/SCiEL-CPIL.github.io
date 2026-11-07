# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Cache for expensive computations to avoid recomputation within a single acquisition call
# We don't need cross-call caching since each call gets fresh mu/sigma

def _safe_norm_cdf(x):
    """Numerically stable norm.cdf."""
    return norm.cdf(x)

def _safe_norm_pdf(x):
    """Numerically stable norm.pdf."""
    return norm.pdf(x)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a blend of Expected Improvement and Uncertainty (UCB-like) 
    weighted by cost efficiency, with careful numerical handling.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 300s limit
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    n = len(mu)
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Clip sigma to avoid division by zero and negative values
    sigma = np.maximum(sigma, 1e-10)
    
    # Clip cost to avoid division by zero
    cost = np.maximum(cost, 1e-10)
    
    # Compute z-score
    improvement = f_best - mu
    z = improvement / sigma
    
    # Numerically stable computation of EI
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For large negative z, EI approaches 0
    # For large positive z, EI approaches (f_best - mu)
    
    # Use np.where to avoid overflow issues, but norm.cdf and norm.pdf are already stable
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement
    ei = improvement * cdf_z + sigma * pdf_z
    
    # Clip EI to non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Uncertainty bonus (UCB-like term)
    # Use a moderate exploration parameter
    beta = 2.0
    ucb_term = beta * sigma
    
    # Blend EI and UCB: use EI as primary, add uncertainty bonus
    # This balances exploitation (EI) and exploration (sigma)
    raw_score = ei + 0.5 * ucb_term
    
    # Cost-aware normalization: divide by cost
    # Higher cost means lower priority
    scores = raw_score / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Clip to reasonable range to avoid extreme values dominating
    # Use a soft cap based on the distribution
    max_score = np.max(scores) if len(scores) > 0 else 1.0
    if max_score > 0:
        scores = np.minimum(scores, 10.0 * max_score)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Return what we have
        pass
    
    return scores
# EVOLVE-BLOCK-END
