"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 315.0  # Safety margin before 330s hard kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Effective Improvement (CEI) acquisition with budget-horizon lookahead.
    
    Structural design: a single acquisition metric where exploration-exploitation
    balance is derived from the GP posterior's own uncertainty structure,
    not from hand-tuned combinations of EI/UCB/PI.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far (lower is better).
    cost : np.ndarray, shape (2000,)
        Evaluation cost for each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = better to evaluate next).
    """
    # Time guard
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        fallback = ei / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # --- CEI with budget-horizon lookahead ---
    
    # Step 1: Compute base Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Step 2: Cost-effective improvement (EI per unit cost)
    cei = ei / cost
    
    # Step 3: Budget-horizon lookahead weight
    # The value of an improvement depends on how much budget remains.
    # We estimate remaining budget pressure from the GP posterior structure:
    # - If mu values are spread out, we're early (exploration phase)
    # - If mu values are concentrated, we're late (exploitation phase)
    
    mu_std = np.std(mu)
    mu_abs_mean = np.mean(np.abs(mu)) + 1e-12
    
    # Temperature: high when the landscape is still uncertain
    # This naturally shifts from exploration to exploitation as the GP
    # becomes more confident (mu_std decreases relative to mu magnitude)
    temperature = 1.0 + 0.5 * (mu_std / mu_abs_mean)
    
    # Step 4: Apply temperature to sigma to modulate exploration
    # Higher temperature -> more weight on uncertain regions
    sigma_eff = sigma * temperature
    
    # Recompute EI with effective sigma
    z_eff = (f_best - mu) / sigma_eff
    ei_eff = (f_best - mu) * norm.cdf(z_eff) + sigma_eff * norm.pdf(z_eff)
    
    # Step 5: Final CEI score
    score = ei_eff / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
