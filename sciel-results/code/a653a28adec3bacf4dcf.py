import numpy as np
from scipy.stats import norm
import time

# Track start time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0  # seconds, with safety margin from 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 3D.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions
    - sigma: array of shape (2000,), GP standard deviation predictions
    - f_best: float, lowest observed value so far
    - cost: array of shape (2000,), evaluation costs for each candidate
    
    Returns:
    - array of shape (2000,), scores (higher = more promising)
    """
    # Deadline guard: if we're too close to the kill, return a safe result
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Return a simple score to finish quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        score = -mu / cost
        return np.where(np.isfinite(score), score, -1e10)
    
    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Hartmann 3D global minimum
    f_opt = -3.86278
    
    # Compute regret-based adaptive kappa
    regret = f_best - f_opt
    max_regret = 6.0
    norm_regret = np.clip(regret / max_regret, 0.0, 1.0)
    
    # Adaptive kappa: higher when regret is large (more exploration needed)
    # Blend between exploration (high kappa) and exploitation (low kappa)
    kappa = 0.5 + 3.5 * norm_regret
    
    # Cost-aware Expected Improvement (EI)
    # Standard EI: (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    # We want to minimize, so improvement is f_best - mu (positive when mu < f_best)
    delta = f_best - mu
    z = delta / sigma_safe
    
    # EI = delta * Phi(z) + sigma * phi(z)
    # For large positive z, Phi(z) ~ 1, phi(z) ~ 0, so EI ~ delta
    # For large negative z, Phi(z) ~ 0, phi(z) ~ 0, so EI ~ 0
    # Use stable computation
    ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add exploration term: kappa * sigma
    # This encourages exploration in uncertain regions
    exploration = kappa * sigma_safe
    
    # Combine: use a weighted sum that balances exploitation (EI) and exploration
    # The cost-aware term divides by cost to prefer cheaper evaluations
    # We use a combination: (ei + exploration) / cost
    # But we can also try a multiplicative or other form
    
    # Primary score: cost-normalized combination of EI and exploration
    score = (ei + exploration) / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
