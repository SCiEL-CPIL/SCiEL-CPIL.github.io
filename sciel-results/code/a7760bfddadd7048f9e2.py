# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Hartmann 3D function definition
def hartmann3d(x):
    """Evaluate Hartmann 3D function at points x (N, 3)."""
    alpha = np.array([1.0, 10.0, 30.0, 3.0, 36.0, 10.0, 30.0, 10.0, 10.0, 10.0,
                      30.0, 3.0, 36.0, 10.0, 10.0, 10.0, 30.0, 3.0, 36.0, 10.0])
    A = np.array([
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0],
        [3.0, 10.0, 30.0],
        [0.1, 10.0, 35.0]
    ])
    P = 2 * np.pi * np.array([
        [36.0, 3.0, 10.0],
        [10.0, 10.0, 35.0],
        [36.0, 3.0, 10.0],
        [10.0, 10.0, 35.0],
        [36.0, 3.0, 10.0],
        [10.0, 10.0, 35.0],
        [36.0, 3.0, 10.0],
        [10.0, 10.0, 35.0],
        [36.0, 3.0, 10.0],
        [10.0, 10.0, 35.0],
        [36.0, 3.0, 10.0],
        [10.0, 10.0, 35.0],
        [36.0, 3.0, 10.0],
        [10.0, 10.0, 35.0],
        [36.0, 3.0, 10.0],
        [10.0, 10.0, 35.0],
        [36.0, 3.0, 10.0],
        [10.0, 10.0, 35.0],
        [36.0, 3.0, 10.0],
        [10.0, 10.0, 35.0]
    ])
    
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    result = np.zeros(x.shape[0])
    for j in range(20):
        term = alpha[j] * np.sum(A[j] * (x**2 - P[j] * x + P[j]**2 / 4.0), axis=1)
        result -= term
    
    return result


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement and Upper Confidence Bound.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : array_like, shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best observed value so far (lowest)
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores, higher means more promising
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure finite values
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) - encourages exploration
    # Beta controls exploration vs exploitation
    beta = 0.5
    ucb = f_best - mu + beta * sigma
    
    # Combine EI and UCB
    # Normalize both to [0, 1] range for better combination
    ei_max = np.max(ei)
    ucb_max = np.max(ucb)
    
    if ei_max > 0:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    if ucb_max > f_best:
        ucb_norm = (ucb - f_best) / (ucb_max - f_best)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Weighted combination: 70% EI, 30% UCB
    scores = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
