"""Rewritten program using Expected Improvement (EI) for acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores using the Expected Improvement (EI) strategy.
    For minimization, EI measures the expected amount by which the function value
    at a candidate point will be lower than the current best value f_best.
    Higher score indicates a better candidate to evaluate next.
    """
    start_time = time.time()
    
    # Safety margin: 280 seconds out of 300 seconds allowed.
    deadline = start_time + 280.0
    
    # Ensure sigma is non-negative and handle zeros to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the improvement: f_best - mu
    improvement = f_best - mu
    
    # Calculate the z-score: (f_best - mu) / sigma
    z = improvement / sigma_safe
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is the CDF and phi is the PDF of the standard normal distribution.
    # Note: Some definitions use -z for minimization depending on sign conventions.
    # Let Y ~ N(mu, sigma^2). We want E[max(0, f_best - Y)].
    # Let Z ~ N(0,1). Y = mu + sigma*Z.
    # f_best - Y = f_best - mu - sigma*Z.
    # Let A = f_best - mu.
    # E[max(0, A - sigma*Z)] = A * P(A - sigma*Z > 0) + E[(A - sigma*Z) * I(A - sigma*Z > 0)]
    # P(A > sigma*Z) = P(Z < A/sigma) = Phi(A/sigma).
    # The integral part results in sigma * phi(A/sigma).
    # So EI = A * Phi(A/sigma) + sigma * phi(A/sigma).
    
    scores = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Check for non-finite values and replace with a very low score (0.0 is neutral, -1e9 is worst)
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    # Deadline guard
    if time.time() > deadline:
        # Return current scores if we are close to timeout
        pass
        
    return scores
# EVOLVE-BLOCK-END
