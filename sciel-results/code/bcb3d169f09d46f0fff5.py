"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for minimizing Griewank 2D.
    
    Uses a hybrid approach: Cost-Effective Expected Improvement with
    adaptive exploration that balances exploitation and exploration
    based on the current state of the GP.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best (lowest) observed value
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    start_time = time.time()
    DEADLINE = 290.0  # Safety margin before 300s limit
    
    # Convert to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # f_best is the best observed value (lowest since we minimize)
    f_best = float(f_best) if np.isfinite(f_best) else 0.0
    
    # Compute Z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip Z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Standard Normal CDF and PDF
    phi_z = norm.cdf(z)  # P(f(x) < f_best)
    pdf_z = norm.pdf(z)  # Standard normal density at z
    
    # Standard EI for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * phi_z + sigma * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration: use a mix of EI and UCB-like term
    # The UCB term (f_best - mu + beta * sigma) encourages exploration
    # We adapt beta based on the current uncertainty level
    
    # Estimate the current exploration-exploitation balance
    # When sigma is high relative to the improvement potential, explore more
    sigma_median = np.median(sigma)
    improvement_potential = np.maximum(f_best - np.median(mu), 1e-9)
    
    # Adaptive beta: higher when uncertainty is high relative to improvement
    beta = 1.0 + 0.5 * (sigma_median / improvement_potential)
    beta = np.clip(beta, 1.0, 3.0)
    
    # UCB-like score
    ucb = (f_best - mu) + beta * sigma
    ucb = np.maximum(ucb, 0.0)
    
    # Hybrid score: blend EI and UCB
    # Weight EI more when we have high confidence (low sigma)
    # Weight UCB more when we have high uncertainty (high sigma)
    confidence = 1.0 / (1.0 + (sigma / sigma_median) ** 2)
    hybrid = confidence * ei + (1.0 - confidence) * ucb
    
    # Cost-aware score: divide by cost
    scores = hybrid / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
