"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline tracking and per-instance counters
_acq_state = {'start_time': None, 'call_count': 0, 'f_best_history': []}

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
    _acq_state['call_count'] += 1
    _acq_state['f_best_history'].append(f_best)
    
    # Deadline guard: if we're getting close to the 330s wall-clock limit,
    # return a safe default to avoid timeout. Use a conservative margin.
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 305.0:
        # Return a flat score to stop exploration safely
        return np.zeros(len(mu), dtype=np.float64)
    
    # Ensure proper array types and handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Ensure sigma is at least a tiny positive value
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # For numerical stability, handle large |z|
    # For z > 8, Phi(z) ~ 1 and phi(z) ~ 0
    # For z < -8, Phi(z) ~ 0 and phi(z) ~ 0
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring with multiple components:
    
    # 1. Base cost-efficiency: EI per unit cost
    base_score = ei / cost
    
    # 2. Smooth exploration weight based on remaining budget
    #    Total budget ~30, start with 5 initial points (cost ~5-10)
    #    Use call count as proxy for remaining budget
    n_calls = _acq_state['call_count']
    # Estimate remaining evaluations: budget 30, each eval costs ~1.5 avg
    # So ~20 total evals possible. We've done n_calls.
    remaining_frac = max(0.0, 1.0 - n_calls / 20.0)
    # Smooth interpolation: early = 0.35, late = 0.05
    exploration_weight = 0.05 + 0.30 * remaining_frac
    
    # 3. Normalized uncertainty component
    #    Scale sigma/cost relative to the current EI landscape
    #    This prevents uncertainty from dominating when EI is already high
    max_ei = np.max(ei) + 1e-10
    uncertainty_score = (sigma / cost) / (max_ei / (np.mean(cost) + 1e-10))
    # Cap to prevent extreme values
    uncertainty_score = np.minimum(uncertainty_score, 2.0)
    
    # 4. Joint exploration-exploitation bonus:
    #    Reward points that are BOTH uncertain AND likely to improve
    #    Use a product term: (sigma/cost) * indicator(mu < f_best)
    #    This encourages exploring in promising regions
    promising_mask = (mu < f_best).astype(np.float64)
    joint_bonus = 0.1 * (sigma / cost) * promising_mask
    joint_bonus = np.minimum(joint_bonus, 0.3 * np.max(joint_bonus) + 1e-10)
    
    # 5. Cost normalization bonus:
    #    Add a small bonus for points with below-average cost
    cost_mean = np.mean(cost)
    cost_std = np.std(cost) + 1e-10
    cost_z = (cost - cost_mean) / cost_std
    # Points with lower cost get a slight bonus
    cost_bonus = 0.06 * (-cost_z)
    cost_bonus = np.clip(cost_bonus, -0.15, 0.15)
    
    # Combine scores:
    # Primary: EI/cost (exploitation)
    # Secondary: normalized sigma/cost (exploration), smoothly weighted
    # Tertiary: joint exploration-exploitation bonus
    # Quaternary: cost efficiency bonus
    scores = base_score * (1.0 + cost_bonus) + exploration_weight * uncertainty_score + joint_bonus
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
