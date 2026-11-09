import numpy as np
from scipy.special import erf
import time

# Global time tracking for deadline guard
_start_time = time.time()
_DEADLINE_OFFSET = 320.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    mu: Array of shape (2000,) - GP mean
    sigma: Array of shape (2000,) - GP standard deviation
    f_best: Float - Best observed value
    cost: Array of shape (2000,) - Cost of evaluation
    
    Returns:
    Array of shape (2000,) - Acquisition scores (higher is better)
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE_OFFSET:
        # Fallback: simple cost-normalized exploit
        mu_safe = np.asarray(mu)
        cost_safe = np.asarray(cost)
        cost_safe = np.maximum(cost_safe, 1e-9)
        return -mu_safe / cost_safe

    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Ensure sigma and cost are positive to avoid division errors
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Numerically stable CDF and PDF using erf
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # phi(z) = exp(-0.5 * z^2) / sqrt(2 * pi)
    
    # For large positive z, erf(z/sqrt(2)) -> 1, so Phi(z) -> 1
    # For large negative z, erf(z/sqrt(2)) -> -1, so Phi(z) -> 0
    # This can suffer from overflow in exp for large |z|
    
    # Clip z to avoid numerical issues in exp
    z_clipped = np.clip(z, -10, 10)
    
    cdf_z = 0.5 * (1.0 + erf(z_clipped / np.sqrt(2.0)))
    pdf_z = np.exp(-0.5 * z_clipped**2) / np.sqrt(2.0 * np.pi)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Adaptive exploration coefficient beta
    # Higher beta encourages exploration (via sigma term)
    # As f_best decreases (progress), we reduce exploration
    # f_best ranges from ~40 (initial) to 0 (optimal)
    # Map f_best to beta: 
    # f_best=40 -> beta=2.5
    # f_best=0 -> beta=0.5
    f_best_clamped = np.clip(f_best, 0.0, 50.0)
    progress = f_best_clamped / 50.0  # 1.0 at start, 0.0 at end
    beta = 0.5 + 2.0 * progress
    
    # Combined Acquisition: Cost-Effective (EI + Beta * Sigma)
    # This balances exploitation (EI) and exploration (Sigma) while penalizing cost
    score = (ei + beta * sigma_safe) / cost_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
