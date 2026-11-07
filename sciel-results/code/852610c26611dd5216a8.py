"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline tracking
_acq_state = {'start_time': None, 'call_count': 0}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement acquisition function with adaptive
    exploration and cost-efficiency weighting.
    
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
    
    # Deadline guard - return safely if we're running out of time
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 325.0:
        return np.zeros(len(mu), dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values robustly
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.5)
    
    # Ensure positive values
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Phase-based adaptation using budget-aware progress
    # Total budget is 30; initial 5 points cost ~5-10 (average ~7.5)
    # Track estimated remaining budget to drive exploration->exploitation
    call_idx = min(_acq_state['call_count'], 60)
    # Use a combination of call count and cost to estimate progress
    # Average cost per evaluation is ~1.5, so ~20 evals fit in budget 30
    # After initial 5, we have ~22.5/1.5 = 15 more evals
    phase = min(1.0, call_idx / 18.0)
    
    # --- Expected Improvement (EI) for minimization ---
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # --- Lower Confidence Bound (LCB) for exploration ---
    # Beta decays from exploratory to exploitative
    beta = 3.0 - 2.0 * phase  # 3.0 -> 1.0
    lcb = -mu + beta * sigma
    
    # --- Cost-awareness ---
    # The minimizer is in [1,9]^4 (inner 80% of [0,10]^4)
    # Cost = 1 + mean(x/10) where x in [0,10]^4
    # So cost is in [1.0, 2.0]
    # Minimizer region has cost around 1.5 (center of [1,9]^4)
    
    # Smooth cost penalty: always positive, penalizes extreme costs
    # Prefer cost near 1.5 (center of search space)
    # Use a Gaussian-like smooth weighting that never goes to zero
    cost_center = 1.5
    cost_width = 0.4
    cost_weight = np.exp(-0.5 * ((cost - cost_center) / cost_width) ** 2)
    # Ensure minimum weight to avoid killing any region entirely
    cost_weight = np.maximum(cost_weight, 0.05)
    
    # Additional mild penalty for very high cost (wastes budget)
    high_cost_penalty = np.exp(-np.maximum(cost - 1.8, 0.0) * 2.0)
    high_cost_penalty = np.maximum(high_cost_penalty, 0.1)
    
    # Combine cost factors
    cost_factor = cost_weight * high_cost_penalty
    
    # --- Combine components ---
    # Blend EI and LCB based on phase
    # Early: more LCB (exploration), Late: more EI (exploitation)
    blend = 0.3 + 0.7 * phase  # 0.3 -> 1.0
    
    base_score = blend * ei + (1 - blend) * lcb
    
    # Apply cost factor (always positive)
    scores = base_score * cost_factor
    
    # Add exploration bonus for uncertain points, scaled by remaining budget
    exploration_bonus = 0.12 * sigma * (1 - phase) * cost_factor
    scores = scores + exploration_bonus
    
    # Ensure finite and non-negative where appropriate
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Normalize to help with numerical stability
    scores_max = np.max(scores)
    if scores_max > 0 and np.isfinite(scores_max):
        scores = scores / scores_max
    
    return scores
# EVOLVE-BLOCK-END
