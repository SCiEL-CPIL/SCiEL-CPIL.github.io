"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline tracking
_acq_state = {'start_time': None, 'call_count': 0}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions in objective units
    - sigma: array of shape (2000,), GP standard deviations in objective units
    - f_best: scalar, lowest observed function value
    - cost: array of shape (2000,), evaluation cost for each candidate
    
    Returns:
    - scores: array of shape (2000,), acquisition values (higher = evaluate next)
    """
    # Initialize module-level state
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
        _acq_state['call_count'] = 0
    
    _acq_state['call_count'] += 1
    
    # Deadline guard
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 320.0:
        return np.zeros(len(mu), dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.5) # Default to mid cost
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement (EI)
    # Minimizing f, so improvement is f_best - f(x)
    z = (f_best - mu) / sigma
    
    # Clip z for numerical stability
    z = np.clip(z, -10, 10)
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # --- Smooth Structural Cost Penalty ---
    # The minimizer is in inner 80%: cost in [1.1, 1.9]
    # Use a smooth Gaussian-like penalty centered at 1.5 with sigma=0.3
    # This smoothly discourages points outside [1.1, 1.9] without hard cutoffs
    
    cost_center = 1.5
    cost_sigma = 0.3
    smooth_penalty = np.exp(-0.5 * ((cost - cost_center) / cost_sigma) ** 2)
    
    # Additional structural insight: cost < 1.1 or cost > 1.9 cannot contain minimizer
    # Apply stronger penalty outside [1.05, 1.95] with smooth transition
    outer_penalty = np.ones_like(cost)
    # Smooth transition from 1.05 to 1.1
    lower_trans = (cost - 1.05) / 0.05
    lower_trans = np.clip(lower_trans, 0, 1)
    outer_penalty = np.minimum(outer_penalty, lower_trans)
    # Smooth transition from 1.95 to 1.9  
    upper_trans = (1.95 - cost) / 0.05
    upper_trans = np.clip(upper_trans, 0, 1)
    outer_penalty = np.minimum(outer_penalty, upper_trans)
    
    # Combine: smooth Gaussian * outer structural constraint
    cost_weight = smooth_penalty * outer_penalty
    cost_weight = np.maximum(cost_weight, 1e-4)  # Keep small floor to avoid complete zeroing
    
    # --- Cost-aware acquisition ---
    # Use EI / sqrt(cost) instead of EI / cost for less aggressive cost penalty
    # This still prefers lower cost but doesn't over-penalize
    
    # Adaptive exploration: early on, explore more; later, exploit
    # call_count tracks how many times acquisition has been called
    call_idx = _acq_state['call_count']
    
    # Decay exploration bonus over time
    # Early: high exploration, Late: low exploration
    exploration_decay = np.exp(-call_idx / 20.0)
    
    # Exploration bonus: sigma-based uncertainty
    exploration_bonus = sigma * exploration_decay * 0.1
    
    # Final score: (EI + exploration_bonus) * cost_weight / sqrt(cost)
    scores = (ei + exploration_bonus) * cost_weight / np.sqrt(cost)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
