"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for minimizing Griewank 2D using a
    structurally different approach: Cost-Aware Expected Improvement with
    uncertainty penalization in the numerator.
    
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
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # But we want to penalize high uncertainty, so we modify this.
    
    # Standard EI
    ei = (f_best - mu) * phi_z + sigma * pdf_z
    
    # Structural change: Penalize uncertainty in the numerator.
    # The idea is that in early, sparse data, high sigma regions are risky.
    # We reduce the EI contribution from the sigma*pdf(z) term when sigma is large.
    
    # A penalty factor that reduces the exploration term based on sigma magnitude.
    # When sigma is small (high confidence), penalty is ~1.
    # When sigma is large (high uncertainty), penalty is smaller.
    # Use a soft penalty: 1 / (1 + (sigma / sigma_scale)^2)
    # sigma_scale controls how quickly the penalty kicks in.
    
    # Estimate a reasonable sigma_scale from the current sigma distribution.
    # Use a robust scale: median of sigma, or a fixed fraction of the range.
    sigma_median = np.median(sigma)
    sigma_scale = max(sigma_median, 1e-6)
    
    # Penalty factor: reduces the sigma*pdf(z) term for high-sigma regions
    penalty = 1.0 / (1.0 + (sigma / sigma_scale) ** 2)
    
    # Modified EI with penalized exploration term
    ei_penalized = (f_best - mu) * phi_z + penalty * sigma * pdf_z
    
    # Ensure EI is non-negative (EI should be >= 0 by definition, but clip for safety)
    ei_penalized = np.maximum(ei_penalized, 0.0)
    
    # Cost-aware score: divide by cost
    scores = ei_penalized / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
