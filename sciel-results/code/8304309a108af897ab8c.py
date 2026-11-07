# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Uses Probability of Improvement (PI) for a distinct exploration bias
    compared to EI, combined with a small UCB component for stability.
    """
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety checks: ensure finite values and avoid division by zero
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    sigma = np.maximum(sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Probability of Improvement for minimization
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    pi = norm.cdf(z)
    
    # Scale PI to be comparable with EI magnitudes
    # EI values can be up to ~f_best, while PI is in [0, 1]
    # Use a scaling factor that adapts to f_best
    scale = np.maximum(f_best, 1e-6)
    pi_scaled = pi * scale
    
    # Small UCB component for additional exploration
    # beta is small to not dominate PI
    beta = 0.1
    ucb = -mu + beta * sigma
    
    # Combine PI and UCB
    # PI provides the main acquisition signal, UCB adds a small exploration push
    score = pi_scaled + 0.1 * ucb
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Deadline guard: if computation took too long, return a safe fallback
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return PI only (fastest path)
        return pi_scaled
    
    return score
# EVOLVE-BLOCK-END
