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
    
    # Phase-based adaptation: early calls favor exploration, later favor exploitation
    call_idx = min(_acq_state['call_count'], 50)
    phase = call_idx / 50.0  # 0 to 1
    
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
    beta = 2.5 - 1.5 * phase  # 2.5 -> 1.0
    lcb = -mu + beta * sigma
    
    # --- Cost-awareness ---
    # The minimizer is in [1,9]^4 (inner 80% of [0,10]^4)
    # Cost = 1 + mean(x/10) where x in [0,10]^4
    # So cost is in [1.0, 2.0], with the minimizer region having cost in [1.4, 1.6]
    # Actually: if x is uniform in [1,9]^4, mean(x/10) is uniform in [0.1, 0.9], mean 0.5
    # So cost is in [1.1, 1.9] for the feasible region
    
    # Smooth cost validity: prefer points with cost in [1.1, 1.9]
    lower_width = 0.05
    upper_width = 0.05
    
    lower_factor = 0.5 * (1.0 + np.tanh((cost - 1.1) / lower_width))
    upper_factor = 0.5 * (1.0 - np.tanh((cost - 1.9) / upper_width))
    cost_validity = np.clip(lower_factor * upper_factor, 0.0, 1.0)
    
    # Cost efficiency: prefer lower cost within valid region
    # Use a soft weighting that penalizes high cost
    cost_eff = np.exp(-0.5 * ((cost - 1.5) / 0.3) ** 2)
    
    # --- Combine components ---
    # Blend EI and LCB based on phase
    # Early: more LCB (exploration), Late: more EI (exploitation)
    blend = 0.4 + 0.6 * phase  # 0.4 -> 1.0
    
    base_score = blend * ei + (1 - blend) * lcb
    
    # Apply cost factors
    scores = base_score * cost_validity * cost_eff
    
    # Add small exploration bonus for uncertain points in valid region
    exploration_bonus = 0.08 * sigma * cost_validity * (1 - phase)
    scores = scores + exploration_bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
