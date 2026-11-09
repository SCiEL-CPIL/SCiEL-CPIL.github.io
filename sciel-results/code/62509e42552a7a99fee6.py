# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) acquisition for Bayesian optimization (minimization).
    
    Structural change: Uses Expected Improvement (EI) instead of Probability of Improvement (PoI).
    EI captures the magnitude of expected regret reduction, which is more informative for 
    minimizing simple regret than just the probability of improvement.
    
    score(x) = EI(x) / cost(x) + beta * sigma(x) / cost(x)
    
    The cost normalization ensures we maximize improvement per unit budget spent.
    The exploration bonus is dynamically scaled:
    - Early in optimization (high f_best), beta is larger to encourage exploration.
    - Late in optimization (low f_best), beta is smaller to focus on exploitation.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions at 2000 candidate points.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions at 2000 candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable).
    """
    t_start = time.time()
    deadline = 280.0  # Safety margin before the 300s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Numerical guards: ensure sigma and cost are positive and finite
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 1e-10), sigma, 1e-6)
    cost_safe = np.where(np.isfinite(cost) & (cost > 1e-10), cost, 1.0)
    
    # Ensure mu is finite
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)

    # Compute Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # For f(x) ~ N(mu, sigma^2):
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    improvement = f_best - mu_safe
    z = improvement / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    z = np.clip(z, -40.0, 40.0)
    
    # Standard normal CDF and PDF
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # EI calculation
    ei = improvement * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical guard)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)

    # Cost-normalized EI: maximize expected improvement per unit cost
    score = ei / cost_safe

    # Adaptive Exploration Bonus
    # We want to explore early when f_best is far from the minimum (0),
    # and exploit later when f_best is close to 0.
    # The three-hump camel 2D function has a global minimum of 0.
    # Initial uniform random points typically yield f_best in [0.5, 2.0].
    # We map f_best to a progress measure [0, 1].
    # progress = 0 when f_best is high (e.g., > 1.5), progress = 1 when f_best is low (<= 0).
    
    # Use a soft normalization. If f_best is negative (shouldn't happen often, but possible due to noise/GP), treat as fully converged.
    if f_best <= 0:
        progress = 1.0
    else:
        # Scale: assume 1.5 is a reasonable "start" value for the scale of the function.
        # As f_best drops from 1.5 to 0, progress goes from 0 to 1.
        progress = np.clip(1.0 - f_best / 1.5, 0.0, 1.0)
    
    # Beta: exploration coefficient
    # Start with moderate exploration (0.8), decrease to 0.32 (0.8 * 0.4) as we converge.
    # This keeps some exploration even late to avoid getting stuck in local minima,
    # but strongly favors exploitation.
    beta = 0.8 * (1.0 - 0.6 * progress)
    
    # Add exploration bonus, also cost-normalized
    exploration_bonus = beta * sigma_safe / cost_safe
    
    score = score + exploration_bonus
    
    # Ensure all scores are finite (higher = evaluate next)
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    if (time.time() - t_start) > deadline:
        return score

    return score
# EVOLVE-BLOCK-END
