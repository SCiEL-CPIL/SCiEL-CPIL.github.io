import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining Expected Improvement (EI) with
    a bounded exploration term.
    
    For 1D sphere minimization, we use a modified EI that balances
    exploitation (finding points where mu is low) with exploration
    (finding points with high uncertainty).
    
    The score is: EI + lambda * sigma, where lambda adapts to f_best.
    When f_best is large (early), we explore more. When f_best is small,
    we exploit more.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-8)
    
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Return a simple fallback to avoid timeout
        return -mu
    
    # Standardized improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is standard normal CDF and phi is standard normal PDF
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    EI = (f_best - mu) * Phi + sigma * phi
    
    # Exploration term: scales with sigma but is bounded
    # Use a smaller weight for exploration as f_best decreases
    # When f_best is large, explore more aggressively
    beta = np.minimum(0.5, 0.1 * np.sqrt(max(f_best, 1e-6)))
    exploration = beta * sigma
    
    # Combined score
    score = EI + exploration
    
    # Handle numerical artifacts
    score = np.nan_to_num(score, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    return score
