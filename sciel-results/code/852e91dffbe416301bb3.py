"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_start_time = time.time()
_TIME_LIMIT = 300.0  # Seconds, as per task contract
_TIME_MARGIN = 5.0   # Safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Expected Improvement (CAEI).
    
    Parameters:
    mu : np.ndarray of shape (2000,) - Mean of GP
    sigma : np.ndarray of shape (2000,) - Std dev of GP
    f_best : float - Best observed value
    cost : np.ndarray of shape (2000,) - Cost of evaluating each point
    
    Returns:
    scores : np.ndarray of shape (2000,) - Acquisition scores (higher is better)
    """
    # Time Guard: If we are close to the limit, return a simple exploration score
    # to guarantee a fast return. This is a fallback; normally EI is fast enough.
    current_time = time.time()
    if current_time - _start_time > _TIME_LIMIT - _TIME_MARGIN:
        # Fallback: Return sigma (exploration) to be safe and fast
        return np.maximum(sigma, 1e-9)
    
    # Ensure inputs are numpy arrays and handle numerical issues
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero or negative values
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard Expected Improvement (EI) for minimization
    # Improvement = f_best - f(x)
    # E[Improvement] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values in PDF/CDF calculation (numerical stability)
    # Phi(10) is effectively 1, phi(10) is effectively 0
    z = np.clip(z, -10, 10)
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI calculation
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (numerical artifacts can cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Aware Adjustment
    # We use a softened cost penalty: cost^0.7
    # This allows the algorithm to pick points with slightly higher cost if they offer significantly higher EI,
    # which helps in escaping local optima more effectively than strict cost-proportionality.
    cost_safe = np.maximum(cost, 1e-9)
    cost_penalty = cost_safe ** 0.7
    
    # Final Score
    scores = ei / cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
