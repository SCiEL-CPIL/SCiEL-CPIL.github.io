# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Uses a hybrid of Expected Improvement (EI) and a scaled uncertainty bonus (sigma)
    to balance exploitation and exploration.
    """
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety checks: ensure finite values and avoid division by zero
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Compute z-score
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)

    # Expected Improvement (EI)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Structural Change: Replace PI with a scaled sigma term.
    # This provides an exploration bonus that scales with the GP uncertainty,
    # encouraging evaluation of points with moderate to high uncertainty
    # if they also have some expected improvement (or are close to f_best).
    # The coefficient 0.5 is a heuristic balance.
    exploration_bonus = 0.5 * sigma
    
    score = ei + exploration_bonus

    # Ensure all scores are finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return a simple exploitation score (negative mean)
        return -mu
    
    return score
# EVOLVE-BLOCK-END
