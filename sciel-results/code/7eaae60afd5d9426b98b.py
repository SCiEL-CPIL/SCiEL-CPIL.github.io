"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 300.0
_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Aware Expected Improvement (CA-EI)
    with a small exploration bonus.
    
    This is a structural change from the previous UCB approach.
    EI integrates the expected improvement over the posterior, providing a
    probabilistically sound measure of utility. Dividing by cost accounts
    for the expense of evaluation. A small exploration bonus (proportional to sigma)
    is added to prevent premature convergence in multimodal landscapes.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising to evaluate next).
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT - _MARGIN:
        # Fallback: return something simple and fast
        # Prefer low cost and high uncertainty
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        return sigma_safe / cost_safe

    # Ensure sigma and cost are positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution
    # Note: If f_best - mu is negative, the term (f_best - mu) * Phi(z) is negative,
    # but phi(z) is positive. The standard formula handles this correctly as EI >= 0.
    # To be safe and efficient, we can compute the components.
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI = max(0, f_best - mu) * Phi(z) + sigma * phi(z)
    # Using max(0, ...) is safer for numerical stability when mu > f_best
    improvement_potential = np.maximum(0, f_best - mu)
    ei = improvement_potential * Phi_z + sigma_safe * phi_z
    
    # Add a small exploration bonus to encourage visiting high-uncertainty regions
    # if they are cheap. This mitigates the greediness of pure EI.
    # The coefficient 0.1 is a heuristic balance between exploitation (EI) and exploration (sigma).
    exploration_bonus = 0.1 * sigma_safe
    
    total_utility = ei + exploration_bonus
    
    # Cost-aware score: utility per unit cost
    scores = total_utility / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
