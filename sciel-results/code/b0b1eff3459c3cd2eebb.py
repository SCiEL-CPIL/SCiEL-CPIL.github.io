"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score using Thompson Sampling with cost normalization.
    
    Args:
        mu: Array of GP means (shape 2000,)
        sigma: Array of GP standard deviations (shape 2000,)
        f_best: Current best observed value (scalar)
        cost: Array of evaluation costs (shape 2000,)
        
    Returns:
        Array of acquisition scores (shape 2000,), higher is better.
    """
    start_time = time.time()
    deadline = 325.0  # Safety margin before 330s limit

    # Convert to numpy arrays and ensure float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Sanitize inputs
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-6)
    cost_safe = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    
    # Thompson Sampling: sample from posterior N(mu, sigma^2)
    # Use a fixed random seed for reproducibility within the call
    rng = np.random.default_rng(42)
    samples = mu_safe + sigma_safe * rng.standard_normal(n)
    
    # Cost-normalized score: lower sampled value per unit cost is better
    # We want to maximize: -sampled_value / cost
    ts_score = -samples / cost_safe
    
    # Add a small exploitation bonus for points where the GP mean is already good
    # This helps when uncertainty is low but the mean is promising
    # Bonus is proportional to how much better the mean is than f_best, normalized by sigma
    improvement = np.maximum(f_best - mu_safe, 0.0)
    # Normalize by sigma to get a z-score-like measure
    z_improvement = improvement / np.maximum(sigma_safe, 1e-9)
    # Cap the z-score to avoid extreme values
    z_improvement = np.clip(z_improvement, 0, 10)
    # Small exploitation bonus (scaled down to not dominate)
    exploit_bonus = 0.1 * z_improvement / cost_safe
    
    # Combine Thompson Sampling score with exploitation bonus
    score = ts_score + exploit_bonus
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check
    if time.time() > deadline:
        return score

    return score
# EVOLVE-BLOCK-END
