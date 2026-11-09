"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Rosenbrock 2d.
    
    Blends Expected Improvement (EI), Probability of Improvement (PI), and
    a quantile-based rank score, all normalized by evaluation cost.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean at candidate points (in objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation at candidate points (in objective units).
    f_best : float
        Best observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    start_time = time.time()
    deadline = 320  # safety margin from 330s limit
    
    # Ensure sigma is positive
    sigma_safe = np.clip(sigma, 1e-10, None)
    
    # Compute z-score: probability that f(x) < f_best
    z = (f_best - mu) / sigma_safe
    
    # Posterior probability of improvement
    p_improve = norm.cdf(z)
    p_improve = np.clip(p_improve, 1e-10, 1.0 - 1e-10)
    
    # Expected Improvement for minimization
    phi_z = norm.pdf(z)
    ei = (f_best - mu) * p_improve + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization
    cost_weight = 1.0 / np.maximum(cost, 1e-10)
    
    # Quantile-based rank: rank candidates by posterior probability of improvement
    # Higher quantile = more likely to beat f_best
    # This is a discrete, robust signal that decouples from noisy mean estimates
    rank = np.argsort(np.argsort(p_improve)) / (len(p_improve) - 1.0)
    rank = np.clip(rank, 0.0, 1.0)
    
    # Combine EI, PI, and quantile rank
    # EI captures magnitude, PI captures confidence, rank provides robust ordering
    score = (0.5 * ei + 0.3 * p_improve + 0.2 * rank) * cost_weight
    
    # Add exploration bonus for high uncertainty regions
    # Use median absolute deviation for robust scale estimation
    mu_median = np.median(mu)
    mad = np.median(np.abs(mu - mu_median)) + 1e-10
    uncertainty_ratio = sigma_safe / (mad + 1e-10)
    uncertainty_ratio = np.clip(uncertainty_ratio, 0, 5.0)
    # Smooth exploration bonus that saturates at high uncertainty
    uncertainty_bonus = 0.1 / (1.0 + uncertainty_ratio)
    
    # Blend main score with uncertainty bonus
    score = 0.85 * score + 0.15 * uncertainty_bonus * cost_weight
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() - start_time > deadline:
        pass  # The computation is already done, just return
    
    return score
# EVOLVE-BLOCK-END
