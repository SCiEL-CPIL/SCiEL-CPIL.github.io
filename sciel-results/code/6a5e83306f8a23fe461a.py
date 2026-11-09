import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using a Thompson Sampling proxy heuristic.
    Maximizes (f_best - mu) / (sigma + epsilon) to balance exploitation
    (low mu) and exploration (high sigma) in a normalized way.
    """
    t_start = time.time()
    deadline = t_start + 280.0

    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling proxy:
    # We want to find points where we expect improvement over f_best.
    # (f_best - mu) is the expected improvement in mean.
    # Dividing by sigma normalizes by uncertainty.
    # High score: high expected improvement relative to uncertainty.
    # This encourages points with low mu (good) and/or high sigma (uncertain),
    # but penalizes high sigma if mu is not sufficiently low.
    
    epsilon = 1e-8
    scores = (f_best - mu) / (sigma + epsilon)
    
    # Check time budget
    if time.time() > deadline:
        pass # Just return what we have
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
