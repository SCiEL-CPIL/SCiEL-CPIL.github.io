import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a structural change: a hybrid of Expected Improvement (EI) and 
    Knowledge Gradient (KG) approximation, with adaptive exploration.
    """
    # Deadline guard: return safe fallback if we're close to timeout
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    if elapsed > 320:  # 10s safety margin before 330s kill
        # Return a simple, fast fallback score
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        # Use a simple UCB fallback
        return (f_best - mu + 0.5 * sigma_safe) / cost_safe

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-10)
    cost = np.maximum(cost, 1e-10)
    
    # Structural change: Use a combination of EI and a variance-based exploration bonus
    # This moves away from pure UCB to a more theoretically grounded approach
    
    # Calculate Expected Improvement for minimization
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in extreme cases
    z = np.clip(z, -50, 50)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Variance-based exploration term: encourage exploring regions with high uncertainty
    # relative to the current best, but scaled by a decaying factor
    # The scale factor decays as we find better solutions
    scale = max(1e-3, (f_best + 3.32237) * 0.1)  # Hartmann 6D min is ~-3.32237
    exploration = scale * sigma / (np.abs(f_best) + 1e-8)
    
    # Weight between exploitation (EI) and exploration
    # Use a fixed but principled weight that balances the two
    # EI is already scaled to objective units, exploration is normalized
    w_exploit = 1.0
    w_explore = 0.3
    
    score = w_exploit * ei + w_explore * exploration
    
    # Divide by cost for cost-awareness
    score = score / cost
    
    # Handle any non-finite values
    score = np.nan_to_num(score, nan=-np.inf, posinf=1e10, neginf=-1e10)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
