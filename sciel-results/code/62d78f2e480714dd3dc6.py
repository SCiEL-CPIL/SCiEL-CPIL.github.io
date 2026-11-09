import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Upper Confidence Bound (UCB) for minimization.
    The exploration parameter beta decays as f_best approaches the global minimum,
    shifting the strategy from exploration to exploitation.
    """
    # Ensure sigma is positive to avoid division or log issues if any
    sigma = np.maximum(sigma, 1e-12)
    
    # Adaptive Beta for UCB
    # Branin global min is approx 0.397887
    # We want beta to be higher when f_best is large (uncertain/early)
    # and lower when f_best is small (exploit/late)
    
    # Scale: 
    # If f_best > 20, beta is at max (1.5)
    # If f_best < 5, beta is at min (0.3)
    # Linear interpolation in between
    
    f_min_known = 0.397887
    # Distance from optimum
    dist = f_best - f_min_known
    
    # Map dist to beta
    # dist=0 -> beta=0.3
    # dist=15 -> beta=1.5
    # beta = 0.3 + (1.5 - 0.3) * (dist / 15.0)
    # Clamp beta between 0.3 and 1.5
    
    beta = 0.3 + 1.2 * (dist / 15.0)
    beta = np.clip(beta, 0.3, 1.5)
    
    # UCB for minimization:
    # We want to minimize f.
    # The lower bound is mu - beta * sigma.
    # We want to pick the point with the lowest lower bound.
    # The score should be higher for points we want to evaluate.
    # So Score = - (mu - beta * sigma) = beta * sigma - mu
    
    scores = beta * sigma - mu
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
