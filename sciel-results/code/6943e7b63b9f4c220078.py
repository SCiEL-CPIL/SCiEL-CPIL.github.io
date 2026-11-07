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
    _acq_state['call_count'] += 1
    
    # Deadline guard: if we're getting close to the 330s wall-clock limit,
    # return a safe default to avoid timeout.
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 310.0:
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
    
    # Bayesian Regret Minimization (BRM) approach:
    # The probability that point x is the global minimum (or close to it)
    # given the GP posterior, weighted by cost efficiency.
    
    # Compute the probability that mu(x) is below f_best (i.e., improvement)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    prob_improve = norm.cdf(z)
    
    # Compute the expected improvement magnitude
    # E[max(f_best - f(x), 0)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * prob_improve + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # BRM-style score:
    # We want to maximize the probability of finding a new minimum,
    # but also account for the magnitude of improvement and cost.
    
    # 1. Primary component: probability of improvement per unit cost
    #    This directly estimates the chance of beating f_best, adjusted for cost.
    prob_per_cost = prob_improve / cost
    
    # 2. Secondary component: expected improvement per unit cost
    #    This captures the magnitude of potential improvement.
    ei_per_cost = ei / cost
    
    # 3. Exploration component: uncertainty per unit cost
    #    In early stages, we should explore more to build the GP model.
    n_calls = _acq_state['call_count']
    # Budget is ~30, initial points cost ~5-10, so ~20 evals remain
    # Early on (n_calls small), weight exploration more heavily
    remaining_frac = max(0.0, 1.0 - n_calls / 25.0)
    exploration_weight = 0.3 * remaining_frac
    
    uncertainty_per_cost = sigma / cost
    
    # 4. Combine: BRM score = w1 * prob/cost + w2 * EI/cost + w3 * sigma/cost
    #    The weights shift from exploration to exploitation over time.
    #    Early: more weight on probability and uncertainty (explore)
    #    Late: more weight on EI (exploit promising regions)
    
    # Adaptive weights based on call count
    if n_calls < 5:
        # Very early: emphasize exploration and probability of improvement
        w_prob = 1.0
        w_ei = 0.3
        w_unc = 0.8
    elif n_calls < 10:
        # Mid: balance all components
        w_prob = 0.8
        w_ei = 0.8
        w_unc = 0.4
    elif n_calls < 15:
        # Later: shift toward exploitation
        w_prob = 0.6
        w_ei = 1.0
        w_unc = 0.2
    else:
        # Late: primarily exploit
        w_prob = 0.4
        w_ei = 1.0
        w_unc = 0.1
    
    scores = w_prob * prob_per_cost + w_ei * ei_per_cost + w_unc * uncertainty_per_cost
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
