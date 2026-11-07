"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Start time for deadline guard
_START_TIME = time.time()
# The evaluator kills after 330s. We want to return before that.
# The acquisition function is called 25 times (30 evals - 5 initial).
# Each call should be fast. 330s / 25 = 13.2s per call.
# EI is very fast (vectorized). We have plenty of margin.
_DEADLINE = _START_TIME + 320 

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian Optimization.
    Uses Expected Improvement (EI) for minimization.
    EI(x) = E[max(0, f_best - f(x))]
    We want to maximize EI.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return random or zeros. Zeros might pick arbitrary.
        # Better to return something that doesn't crash.
        return np.zeros_like(mu)

    # Ensure sigma is strictly positive to avoid division by zero in z calculation
    # and to avoid numerical issues in the normal pdf/cdf.
    # 1e-8 is a safe small number.
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score: (f_best - mu) / sigma
    # If mu < f_best, z > 0. Improvement is likely.
    # If mu > f_best, z < 0. Improvement is unlikely.
    z = (f_best - mu) / sigma_safe
    
    # Calculate Probability of Improvement
    # PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # Where Phi is the CDF of the standard normal distribution.
    # This aggressively exploits regions where the probability of improvement is high.
    pi = norm.cdf(z)
    
    # Ensure finite scores
    # PI should be in [0, 1].
    pi = np.clip(pi, 0.0, 1.0)
    
    # Replace any remaining non-finite values
    pi = np.nan_to_num(pi, nan=0.0, posinf=1.0, neginf=0.0)
    
    return pi
# EVOLVE-BLOCK-END
