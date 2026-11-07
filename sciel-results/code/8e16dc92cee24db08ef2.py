"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline tracking
_acq_state = {'start_time': None, 'call_count': 0}

def acquisition(mu, sigma, f_best, cost):
    """
    Bayesian Regret Minimization (BRM) acquisition function.
    
    Instead of pointwise EI, we approximate the probability that each candidate
    is the global minimum given the remaining budget, weighted by cost efficiency.
    
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
    
    # Deadline guard
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 305.0:
        return np.zeros(len(mu), dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    n = len(mu)
    
    # BRM approach: approximate the probability that candidate i is the global min
    # within the remaining budget. We use a sampling-free approximation:
    # P(i is global min) ≈ exp(-λ * (mu_i - mu_min) / sigma_i) * exp(-μ * cost_i)
    # where λ controls how strongly we prefer lower mu, and μ controls cost penalty.
    
    # Normalize mu and sigma for stability
    mu_min = np.min(mu)
    mu_range = np.max(mu) - mu_min + 1e-10
    sigma_max = np.max(sigma) + 1e-10
    
    # Probability of being near the global minimum (exploitation signal)
    # Use a soft-min approximation: points with low mu and low sigma are favored
    z_exploit = (mu - mu_min) / (sigma + 1e-10)
    # P(low value) = Phi(-z_exploit) — probability that f(x_i) < mu_min
    prob_low = norm.cdf(-z_exploit)
    
    # Uncertainty bonus: reward points with high sigma (exploration)
    # but only if they're not too far from the current best
    uncertainty_bonus = sigma / (sigma_max + 1e-10)
    
    # Cost efficiency: prefer lower cost
    cost_mean = np.mean(cost)
    cost_std = np.std(cost) + 1e-10
    cost_z = (cost - cost_mean) / cost_std
    cost_efficiency = np.exp(-0.3 * cost_z)  # Lower cost → higher efficiency
    
    # Combine: BRM score = P(global min) * cost_efficiency + exploration_weight * uncertainty
    # The exploration weight decays as we approach the budget limit
    n_calls = _acq_state['call_count']
    remaining_frac = max(0.0, 1.0 - n_calls / 18.0)
    exploration_weight = 0.15 + 0.25 * remaining_frac
    
    scores = prob_low * cost_efficiency + exploration_weight * uncertainty_bonus * (1.0 - prob_low)
    
    # Ensure finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
