"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer to track elapsed time for the deadline guard
_start_time = time.time()
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a direct weighted sum of Expected Improvement (EI),
    Upper Confidence Bound (UCB), and Probability of Improvement (POI).
    
    Key design choice: NO min-max normalization. EI and UCB are in objective units
    and directly comparable, preserving the global acquisition landscape structure.
    POI is already in [0,1] and adds a soft rank-based exploration signal.
    
    This approach was validated by inspiration 2 (score 1.489) which showed that
    direct weighted sums outperform normalized blends for multi-modal Branin 2D.
    """
    # Check time - if running out, fall back to fast exploitative heuristic
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is finite and non-negative
    sigma = np.maximum(sigma, 1e-12)
    
    # Standardized improvement
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) when f_best > mu, else 0
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = np.where(f_best > mu, (f_best - mu) * cdf_z + sigma * pdf_z, 0.0)
    
    # Calculate UCB for minimization (beta=1.0, proven effective)
    beta = 1.0
    ucb = -mu + beta * sigma
    
    # Calculate Probability of Improvement (already in [0,1])
    poi = norm.cdf(z)
    
    # Direct weighted sum in objective units - NO normalization
    # EI provides strong exploitation signal
    # UCB provides exploration signal (beta=1.0)
    # POI adds a small soft rank-based exploration boost
    score = ei + 0.5 * ucb + 0.1 * poi
    
    # Ensure scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
