# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track total evaluations for adaptive exploration
_total_cost_accumulated = 0.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a hybrid of cost-aware EI, adaptive LCB, and confidence-weighted bonus.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - higher values are preferred
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 295.0  # 5s safety margin
    
    # Ensure arrays are 1D and float
    mu = np.asarray(mu, dtype=np.float64).flatten()
    sigma = np.asarray(sigma, dtype=np.float64).flatten()
    cost = np.asarray(cost, dtype=np.float64).flatten()
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    score_ei = ei / cost_safe
    
    # Probability of Significant Improvement (PSI)
    delta = max(0.01 * abs(f_best), 1e-6)
    z_psi = (f_best - delta - mu) / sigma_safe
    psi = norm.cdf(z_psi)
    
    # Cost-aware PSI
    psi_per_cost = psi / cost_safe
    
    # Adaptive exploration coefficient that decays with total cost consumed
    remaining_budget_fraction = max(0.0, 1.0 - _total_cost_accumulated / 30.0)
    
    # Lower Confidence Bound (LCB) component
    base_beta = 2.0
    epsilon = 0.1
    beta = base_beta * (remaining_budget_fraction + epsilon)
    
    lcb = mu - beta * sigma_safe
    lcb_per_cost = -lcb / cost_safe  # Negate because we want higher scores for lower LCB
    
    # Confidence-weighted bonus: encourages exploitation of high-confidence promising regions
    # This bonus is active when mu is much better than f_best and sigma is small
    confidence_mask = (f_best - mu) > 0.1 * f_best
    confidence_bonus = np.where(confidence_mask, 
                                 (f_best - mu) / (sigma_safe + 1e-9) / cost_safe, 
                                 0.0)
    confidence_bonus = np.maximum(confidence_bonus, 0.0)
    
    # Normalize components to comparable scale
    max_ei = np.max(score_ei)
    if max_ei > 1e-12:
        ei_norm = score_ei / max_ei
    else:
        ei_norm = score_ei
    
    max_psi = np.max(psi_per_cost)
    if max_psi > 1e-12:
        psi_norm = psi_per_cost / max_psi
    else:
        psi_norm = psi_per_cost
    
    max_lcb = np.max(lcb_per_cost)
    if max_lcb > 1e-12:
        lcb_norm = lcb_per_cost / max_lcb
    else:
        lcb_norm = lcb_per_cost
    
    max_conf = np.max(confidence_bonus)
    if max_conf > 1e-12:
        conf_norm = confidence_bonus / max_conf
    else:
        conf_norm = confidence_bonus
    
    # Dynamic weights:
    # - When f_best is high (early), weight LCB and PSI more for exploration
    # - When f_best is low (late), weight EI and confidence bonus more for exploitation
    f_best_abs = abs(f_best)
    
    # Weight for EI (exploitation): increases as f_best decreases
    w_ei = 0.4 + 0.4 * (1.0 / (1.0 + f_best_abs / 2.0))
    
    # Weight for LCB (exploration via uncertainty): decreases as budget is consumed
    w_lcb = 0.3 * remaining_budget_fraction
    
    # Weight for PSI (exploration via probability of improvement): moderate
    w_psi = 0.2 * (1.0 - 0.5 * remaining_budget_fraction)
    
    # Weight for confidence bonus: increases as f_best decreases
    w_conf = 0.3 * (1.0 / (1.0 + f_best_abs / 2.0))
    
    # Normalize weights to sum to 1
    w_sum = w_ei + w_lcb + w_psi + w_conf
    if w_sum < 1e-9:
        w_sum = 1e-9
    w_ei /= w_sum
    w_lcb /= w_sum
    w_psi /= w_sum
    w_conf /= w_sum
    
    # Combine components
    scores = w_ei * ei_norm + w_lcb * lcb_norm + w_psi * psi_norm + w_conf * conf_norm
    
    # Check time
    if time.time() > deadline:
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
