"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 315.0  # Safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) per unit cost.
    
    This function computes the Expected Improvement for each candidate point
    based on the GP posterior mean (mu) and standard deviation (sigma),
    and normalizes it by the evaluation cost.
    
    Parameters:
    mu (np.ndarray): GP posterior means, shape (N,)
    sigma (np.ndarray): GP posterior standard deviations, shape (N,)
    f_best (float): Best observed function value so far.
    cost (np.ndarray): Evaluation costs for each candidate, shape (N,)
    
    Returns:
    np.ndarray: Acquisition scores, shape (N,). Higher scores indicate
                candidates that are more likely to reduce the regret
                per unit cost.
    """
    # Deadline guard: if we are running out of time, return a safe fallback
    # Returning zeros is safe because the evaluator picks the max, and if all are 0,
    # it might pick index 0 or random. However, to be safer against invalid states,
    # we return the previous best logic if possible, but we don't have state.
    # Zeros are finite and valid.
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)

    # Convert to numpy arrays for consistent typing
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-zero to avoid division by zero in z-score
    # A very small epsilon prevents numerical instability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Ensure cost is non-zero
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate the z-score: how many standard deviations f_best is above mu
    # z > 0 means mu < f_best, so there is potential for improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # For minimization, EI is the expected amount by which f(x) improves f_best
    # EI = (f_best - mu) * P(f(x) < f_best) + sigma * E[f(x) | f(x) < f_best]
    # Using the standard normal CDF (Phi) and PDF (phi):
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Use scipy.stats for stable computation
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Cost-aware score: EI divided by cost
    # This encourages both high improvement and low cost
    scores = ei / cost_safe
    
    # Replace any non-finite values with 0.0 to ensure validity
    # This handles any potential numerical overflow or NaNs from extreme inputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
