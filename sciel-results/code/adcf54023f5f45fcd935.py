import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0

def _deadline_reached():
    return time.time() - _start_time > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Cost-Adjusted Expected Hypervolume Improvement (EHI) with tanh saturation.
    
    mu: array of shape (2000,) - GP mean in objective units
    sigma: array of shape (2000,) - GP std dev in objective units
    f_best: float - best observed value so far
    cost: array of shape (2000,) - evaluation cost for each point, 1 + mean(x)
    
    Returns: array of shape (2000,) with acquisition scores (higher = better)
    """
    if _deadline_reached():
        return np.zeros_like(mu)

    # Ensure finite inputs
    mu = np.nan_to_num(mu, nan=f_best, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement for minimization
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    from scipy.stats import norm
    
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    # EI for minimization: improvement = f_best - f(x)
    # E[improvement] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost adjustment: divide by cost to prefer cheaper points
    # Use tanh saturation to avoid over-exploration of high-cost, high-uncertainty regions
    # The key insight: EHI with tanh properly balances improvement vs uncertainty
    # and discriminates between cheap high-improvement and expensive low-improvement
    
    # Normalize cost to [0, 1] range for better scaling
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-12
    cost_norm = (cost - cost_min) / cost_range
    
    # Cost-aware EHI: use tanh to saturate the improvement signal
    # This prevents linear over-exploration of high-cost regions
    # The tanh term ensures that very large improvements don't dominate excessively
    
    # Scale EI relative to typical improvement scale
    # Use a characteristic scale based on current uncertainty
    sigma_mean = np.mean(sigma)
    scale = max(sigma_mean, 0.1)
    
    # EHI-like score: tanh(ei / (scale * cost_factor)) / cost
    # cost_factor makes expensive points require larger improvements to be attractive
    cost_factor = 1.0 + 2.0 * cost_norm
    
    # Apply tanh saturation to balance improvement against uncertainty
    raw_score = ei / (scale * cost_factor)
    saturated = np.tanh(raw_score)
    
    # Final score: saturated improvement divided by cost
    # This creates a nonlinear preference for cheap points with good expected improvement
    score = saturated / (cost * 0.5 + 0.5)
    
    # Add a small exploration bonus for high uncertainty points
    # This helps escape local optima but is damped by cost
    exploration_bonus = 0.3 * sigma / (cost * 0.5 + 0.5)
    
    score = score + exploration_bonus
    
    # Ensure all scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
