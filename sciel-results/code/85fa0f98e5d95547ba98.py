"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition blending Expected Improvement (EI) with
    probability of improvement (PI) and a small uncertainty bonus.
    
    The core signal is EI (captures expected reduction in objective value),
    augmented by PI (robust probability of beating f_best) and a mild
    exploration term for high-uncertainty regions. All are normalized by
    cost to favor cheap evaluations.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean at candidate points (in objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation at candidate points (in objective units).
    f_best : float
        Best observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    start_time = time.time()
    deadline = 320  # safety margin from 330s limit
    
    # Structural change: Quantile-based cost-aware acquisition
    # Instead of continuous EI/PI, use the predictive distribution's
    # lower quantile as a robust "expected best value" estimate,
    # then rank by cost-normalized quantile.
    
    # Ensure sigma is positive
    sigma_safe = np.clip(sigma, 1e-10, None)
    
    # Compute the 10th percentile of the predictive normal distribution
    # This is a robust estimate of the "expected minimum" at each point
    # z_0.10 = -1.2816 (standard normal quantile for 10th percentile)
    z_quantile = -1.2816
    quantile_estimate = mu + z_quantile * sigma_safe
    
    # The "improvement potential" is how much better this quantile
    # is compared to f_best (lower is better for the objective)
    # We want points where the quantile is much lower than f_best
    improvement_potential = f_best - quantile_estimate
    improvement_potential = np.maximum(improvement_potential, 0.0)
    
    # Cost normalization: divide by cost to favor cheap evaluations
    # Use a power of cost to control the strength of cost preference
    # cost^1 gives linear cost penalty
    cost_denom = cost
    
    # Primary score: cost-normalized improvement potential
    score = improvement_potential / cost_denom
    
    # Add a small uncertainty bonus for exploration in high-sigma regions
    # This helps escape local optima in the Rosenbrock valley
    # Normalize sigma to [0, 1] range for the bonus
    sigma_norm = sigma_safe / (sigma_safe.max() + 1e-10)
    exploration_bonus = 0.05 * sigma_norm / cost_denom
    
    score = score + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    if time.time() - start_time > deadline:
        pass  # The computation is already done, just return
    
    return score
# EVOLVE-BLOCK-END
