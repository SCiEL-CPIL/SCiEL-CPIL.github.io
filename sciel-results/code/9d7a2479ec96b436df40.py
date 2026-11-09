"""Cost-aware EI acquisition for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function.
    
    score = EI(x) / cost(x) + beta * sigma(x) / cost(x)
    
    EI is the expected improvement over f_best for minimization.
    The cost normalization makes the acquisition cost-aware.
    The exploration bonus (beta * sigma / cost) encourages exploration
    in uncertain regions, scaled by the same cost factor.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable).
    """
    t0 = time.time()
    deadline = 280.0  # Safety margin before 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure finite and positive values
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-10), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 1e-10), cost, 1.0)

    # Compute Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For GP: f(x) ~ N(mu, sigma^2)
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    # Handle the case where f_best might be very small or zero
    improvement = f_best - mu  # positive when mu < f_best (potential improvement)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    z = improvement / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # EI formula for minimization
    ei = improvement * Phi + sigma_safe * phi
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI
    score = ei / cost
    
    # Exploration bonus: encourage exploration in uncertain regions
    # Scale beta based on how close we are to convergence
    # When f_best is large (early), use more exploration
    # When f_best is small (late), use less exploration
    
    # Estimate progress: f_best starts around 1-2 and approaches 0
    # Use a soft normalization
    if f_best > 0:
        progress = np.clip(1.0 - f_best / 1.5, 0.0, 1.0)
    else:
        progress = 1.0
    
    # Beta: exploration coefficient
    # Start with moderate exploration, decrease as we converge
    beta = 0.8 * (1.0 - 0.6 * progress)
    
    # Add exploration bonus, also cost-normalized
    exploration_bonus = beta * sigma_safe / cost
    
    score = score + exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check time
    if time.time() - t0 > deadline:
        return score
    
    return score
# EVOLVE-BLOCK-END
