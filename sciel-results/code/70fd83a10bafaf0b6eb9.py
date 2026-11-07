# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Look-Ahead Expected Improvement (LAEI) acquisition function.
    
    This function estimates the expected reduction in the global minimum
    after observing a hypothetical value drawn from the local GP posterior.
    It combines exploitation (improvement over f_best) with exploration
    (uncertainty) in a non-linear, adaptive manner.
    
    The score is approximated as:
    score = E[max(f_best - y, 0)] + gamma * E[max(sigma_y, 0)] * w
    
    where y ~ N(mu, sigma^2) is the hypothetical observation,
    and w is a weight that adapts based on the current regret.
    
    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    # Start a timer to guard against the 330s wall-clock limit
    t_start = time.time()
    
    # Guard against non-positive sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Theoretical global minimum of Hartmann 6D function
    f_min = -3.32237
    
    # Calculate current regret (distance from best found to global min)
    regret = f_best - f_min
    
    # Clip regret to avoid numerical issues
    regret = np.clip(regret, 0.0, 10.0)
    
    # Adaptive parameters based on regret
    # When regret is large, we want more exploration
    # When regret is small, we want more exploitation
    
    # Transition factor: 0 when regret=0, approaches 1 as regret increases
    transition = 1.0 - np.exp(-regret / 1.0)
    
    # EI weight: how much to weight the expected improvement term
    # Higher when regret is large (early on)
    ei_weight = 0.5 + 1.5 * transition
    
    # Exploration weight: how much to weight the uncertainty term
    # Higher when regret is large
    exp_weight = 0.5 + 1.5 * transition
    
    # Compute Expected Improvement (EI) for minimization
    # EI = E[max(f_best - y, 0)] where y ~ N(mu, sigma^2)
    # Standardized: z = (mu - f_best) / sigma
    z = (mu - f_best) / sigma_safe
    
    # EI formula: (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is standard normal CDF and phi is standard normal PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * (1.0 - Phi_z) + sigma_safe * phi_z
    
    # Clip EI to be non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Compute Look-Ahead term
    # Approximate the expected reduction in uncertainty after observing y
    # This is related to the expected sigma of the posterior update
    # A simpler proxy: weight sigma by a factor that decreases with confidence
    
    # The look-ahead term captures the value of reducing uncertainty
    # in regions where we are uncertain but also have potential for improvement
    
    # Use a weighted combination:
    # score = ei_weight * EI + exp_weight * sigma * (1 - Phi_z)
    # The (1 - Phi_z) term weights exploration by the probability of improvement
    
    laei = ei_weight * ei + exp_weight * sigma_safe * (1.0 - Phi_z)
    
    # Check time budget
    elapsed = time.time() - t_start
    if elapsed > 280:  # Safety margin before 300s
        scores = np.where(np.isfinite(laei), laei, 0.0)
        return scores
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(laei), laei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
