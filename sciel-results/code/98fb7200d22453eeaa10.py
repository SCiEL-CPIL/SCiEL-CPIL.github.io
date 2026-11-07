import numpy as np
import time
from scipy.special import erf, expit

# Constants for EI calculation
SQRT_2PI = np.sqrt(2 * np.pi)
INV_SQRT_2 = 1 / np.sqrt(2)

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) with an entropy-based beta.
    """
    t_start = time.time()
    deadline = t_start + 0.050
    
    epsilon = 1e-10
    sigma_safe = np.maximum(sigma, epsilon)
    
    # Structural Change: Use UCB with beta derived from the number of candidates (N=2000)
    # beta = sqrt(log(N))
    # This is a principled choice for balancing exploration/exploitation in the candidate set.
    beta = np.sqrt(np.log(2000))
    
    # UCB for minimization: mu - beta * sigma
    # We want to MINIMIZE f, so we pick the point with the lowest "lower bound".
    # The acquisition score should be HIGH for points we want to pick.
    # So Score = - (mu - beta * sigma) = -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
