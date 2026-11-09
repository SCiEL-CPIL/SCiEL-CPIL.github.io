"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Deadline guard: return a simple fallback if we're getting close to timeout
    deadline_margin = 280  # seconds margin from 300s limit
    
    # Ensure inputs are valid
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute Phi(z) and phi(z) using numpy for speed
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # phi(z) = exp(-z^2 / 2) / sqrt(2*pi)
    pi = np.pi
    sqrt_2pi = np.sqrt(2 * pi)
    sqrt_2 = np.sqrt(2)
    
    # Use numpy's erf approximation via scipy if available, otherwise use a fast approximation
    # For speed, we'll use a vectorized computation
    try:
        from scipy.special import erf, erfc
        Phi_z = 0.5 * (1 + erf(z / sqrt_2))
        phi_z = np.exp(-0.5 * z * z) / sqrt_2pi
    except ImportError:
        # Fallback: use a simple approximation
        Phi_z = 0.5 * (1 + np.tanh(z / sqrt_2))  # Approximation
        phi_z = np.exp(-0.5 * z * z) / sqrt_2pi
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Cost-normalized EI
    ei_cost = ei / cost_safe
    
    # Compute a quantile-based heuristic:
    # Use the probability that the point improves over f_best, weighted by expected gain
    # This is essentially PI (Probability of Improvement) = Phi(z)
    pi_score = Phi_z
    
    # Blend: primarily cost-normalized EI with some PI for exploration
    # The key insight: cost-aware acquisition should favor points with high EI per unit cost
    # but also account for the probability of actual improvement
    
    # Use a weighted combination that emphasizes cost efficiency
    # Weight: 0.7 for EI/cost, 0.3 for PI/cost
    score = 0.7 * ei_cost + 0.3 * pi_score / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline_margin:
        # Return EI cost as fallback
        ei_cost = np.where(np.isfinite(ei_cost), ei_cost, 0.0)
        return ei_cost
    
    return score
# EVOLVE-BLOCK-END
