"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Adaptive Lower Confidence Bound (LCB) / UCB acquisition function.
    
    Instead of Expected Improvement, this uses a linear combination of the
    predicted mean and standard deviation: Score = beta * sigma - mu.
    We want to minimize the function, so we maximize the score which 
    corresponds to minimizing (mu - beta * sigma).
    
    The parameter beta is adapted dynamically based on the median uncertainty
    of the candidate set. This allows the strategy to start with high exploration
    (large beta relative to sigma) and shift to exploitation as uncertainty drops.
    """
    # Deadline guard: although this function is fast, we ensure we don't hang
    # if numpy operations somehow stall (unlikely but safe).
    start_time = time.time()
    deadline = start_time + 250.0  # 30s margin before 280s limit? 
    # The evaluator kills at 330s. The loop has 30 evals. 
    # This function is called once per eval. 30 * (time per call) < 330.
    # If one call takes > 10s, we might be in trouble. 
    # Numpy ops on 2000 floats are microsecond scale. This is safe.
    
    # Handle edge cases where sigma is all zeros or very small
    if np.all(sigma < 1e-10):
        # If no uncertainty, just exploit: pick lowest mu
        # Score should be high for low mu. 
        # Return -mu
        return -mu
    
    # Calculate adaptive beta
    # Using median sigma as a scale factor ensures beta scales with the "temperature" of the GP
    # kappa controls the exploration-exploitation trade-off
    # kappa=1.0 means exploration term is comparable to typical uncertainty
    # kappa=2.0 means stronger exploration
    kappa = 1.5
    
    sigma_median = np.median(sigma)
    beta = kappa * sigma_median
    
    # Compute LCB score: we want to minimize f, so we look for low (mu - beta*sigma)
    # Maximizing score = beta*sigma - mu is equivalent
    score = beta * sigma - mu
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
