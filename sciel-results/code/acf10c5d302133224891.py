import numpy as np
import time
from scipy.stats import norm

# Global state for timing
_acq_state = {
    'start_time': None
}

# Adaptive parameters for cost-aware UCB/EI hybrid
_acq_params = {
    'beta_ucb': 2.0,
    'beta_ei': 1.0,
    'exploration_decay': 0.5,
    'cost_norm': 1.0
}

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 285:
        # Fallback: simple LCB with cost normalization
        eps = 1e-9
        s = np.maximum(sigma, eps)
        c = np.maximum(cost, eps)
        score = -(mu + 2.0 * s) / c
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Cost-Aware Hybrid: UCB + EI with adaptive exploration
    # mu, sigma, cost are arrays of shape (2000,)
    # f_best is a scalar
    
    eps = 1e-9
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)
    
    # Compute total cost to determine budget progress
    total_cost = np.sum(cost)
    
    # Budget is 30, so progress goes from 0 to 1 as cost approaches 30
    # Clamp to [0, 1]
    budget_progress = np.clip(total_cost / 30.0, 0.0, 1.0)
    
    # Adaptive UCB beta: higher early for exploration, lower late for exploitation
    # beta_ucb starts at 2.0 and decays to 1.0 as budget is consumed
    beta_ucb = _acq_params['beta_ucb'] - (_acq_params['beta_ucb'] - 1.0) * budget_progress
    
    # EI weight: starts lower, increases as we get more information
    # This shifts from UCB-driven exploration to EI-driven exploitation
    beta_ei = _acq_params['beta_ei'] * (0.3 + 0.7 * budget_progress)
    
    # Compute UCB score: -mu + beta * sigma (for minimization, lower is better)
    ucb_score = -mu + beta_ucb * s
    
    # Compute EI for minimization
    z = (f_best - mu) / s
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + s * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Normalize costs: divide by mean cost to get relative cost
    mean_cost = np.mean(c)
    c_norm = c / (mean_cost + eps)
    
    # Hybrid score: combine UCB and EI, both normalized by cost
    # UCB term encourages exploring high-uncertainty regions
    # EI term focuses on regions likely to improve f_best
    # Both are divided by cost to prefer cheaper evaluations
    score_ucb = ucb_score / c_norm
    score_ei = ei / c_norm
    
    # Weighted combination
    score = beta_ei * score_ei + (1.0 - beta_ei) * score_ucb
    
    # Add a small exploration floor for very uncertain, cheap regions
    # This helps escape local minima in early iterations
    exploration_weight = _acq_params['exploration_decay'] * (1.0 - budget_progress)
    exploration_floor = exploration_weight * s / c_norm
    
    score = score + exploration_floor
    
    # Clean up
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
