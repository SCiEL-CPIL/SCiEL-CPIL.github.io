"""Acquisition function for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for minimizing sphere 1d on [-5, 5].
    Uses an adaptive Lower Confidence Bound (LCB) strategy.
    For minimization, we maximize the score: -(mu - beta * sigma) = -mu + beta * sigma.
    Beta is adapted based on the current best value f_best.
    """
    # Clip sigma to avoid numerical issues, though it should be positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # The sphere function f(x) = x^2 on [-5, 5] has a maximum value of 25.
    # We use this to normalize the progress of the optimization.
    f_max_possible = 25.0
    
    # Calculate progress: 0 at start (f_best ~ 25), 1 at end (f_best ~ 0)
    # Clip f_best to [0, f_max_possible] to be safe
    f_best_clipped = np.clip(f_best, 0.0, f_max_possible)
    progress = 1.0 - (f_best_clipped / f_max_possible)
    
    # Adaptive Beta (exploration parameter)
    # Start with high exploration (beta_high) and reduce to low exploration (beta_low)
    # as we get closer to the minimum.
    # Use a smooth decay for better behavior
    beta_high = 3.0
    beta_low = 0.3
    
    # Use exponential decay for smoother transition
    # beta = beta_low + (beta_high - beta_low) * exp(-k * progress)
    # At progress=0: beta = beta_high
    # At progress=1: beta = beta_low + (beta_high - beta_low) * exp(-k)
    # Choose k so that at progress=1, beta is close to beta_low
    k = 3.0
    beta = beta_low + (beta_high - beta_low) * np.exp(-k * progress)
    
    # LCB score for minimization:
    # We want to find points that might have low true values.
    # The lower bound of the true value is roughly mu - beta * sigma.
    # To select the best point, we maximize the "potential for improvement" or minimize the lower bound.
    # Score = - (mu - beta * sigma) = -mu + beta * sigma
    score = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
