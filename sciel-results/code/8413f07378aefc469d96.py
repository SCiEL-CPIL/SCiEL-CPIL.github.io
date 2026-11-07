# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization using an ensemble approach.
    Returns 2000 finite scores; higher is better.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s limit
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    
    # Clip sigma to a small positive value
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-scores
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Ensemble of acquisition functions
    
    # 1. Expected Improvement (EI) for minimization
    ei = diff * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Upper Confidence Bound (UCB) for minimization
    # UCB = mu - kappa * sigma (we want low values, so subtract exploration term)
    # Adaptive kappa based on confidence
    kappa = 0.1  # Moderate exploration
    ucb = mu - kappa * sigma_safe
    
    # 3. Probability of Improvement (PI) for minimization
    # PI = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)
    
    # 4. Normalized versions for fair combination
    # Normalize each to [0, 1] range
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-12:
            return np.ones_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb)
    pi_norm = normalize(pi)
    
    # 5. Weighted ensemble - adapt weights based on optimization stage
    # Early: more exploration (UCB), Late: more exploitation (EI)
    # Use f_best as a proxy for progress (lower = better progress)
    
    # Check time budget
    if time.time() > deadline - 10.0:
        # Return simple EI if running out of time
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei
    
    # Adaptive weights: start with balanced, shift toward exploitation as f_best improves
    # Ackley minimum is 0, so f_best close to 0 means we're doing well
    if f_best < 0.1:
        # Late stage: exploit more
        w_ei, w_ucb, w_pi = 0.6, 0.2, 0.2
    elif f_best < 1.0:
        # Mid stage: balanced
        w_ei, w_ucb, w_pi = 0.4, 0.3, 0.3
    else:
        # Early stage: explore more
        w_ei, w_ucb, w_pi = 0.3, 0.4, 0.3
    
    # Combined score
    ensemble_score = w_ei * ei_norm + w_ucb * ucb_norm + w_pi * pi_norm
    
    # Ensure finite scores
    ensemble_score = np.where(np.isfinite(ensemble_score), ensemble_score, 0.0)
    
    return ensemble_score
# EVOLVE-BLOCK-END
