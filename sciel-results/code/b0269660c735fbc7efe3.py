"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function with dynamic exploration constant.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : np.ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed value so far (lowest since we minimize).
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        UCB scores; higher values indicate more promising candidates.
    """
    # Estimate number of evaluations from the context.
    # Since we don't have direct access to the evaluation count, we infer it
    # from the relationship between f_best and the known minimum.
    # However, a more robust approach: use a fixed schedule based on typical BO settings.
    # For 30 total evaluations with 5 initial, we have ~25 BO steps.
    # We'll use a heuristic: kappa increases with sqrt(n_evals).
    
    # Since we can't track n_evals directly in a stateless function, we use
    # a principled static schedule. The UCB parameter beta_t is typically:
    # beta_t = log(t^2 * pi^2 / (6 * delta)) for theoretical guarantees
    # For practical purposes, we use a simpler increasing schedule.
    
    # Given the task has 30 evaluations total (5 initial + 25 BO steps),
    # and each acquisition call corresponds to one BO step, we can estimate
    # the current step by looking at how close f_best is to the optimum.
    
    # Alternative: use a fixed but well-tuned kappa that works across the range.
    # From literature, kappa in range [1, 5] works well for 3D problems.
    # We'll use a dynamic approach: start with higher exploration, decrease over time.
    
    # Since the function is stateless, we need a proxy for the number of evaluations.
    # One approach: the spread of sigma values can indicate how much exploration has happened.
    # Early on, sigma is large everywhere; later, it's small except in unexplored regions.
    
    # Compute a proxy for exploration level based on sigma statistics
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    
    # If sigma is large, we're early in the optimization -> more exploration
    # If sigma is small, we're later -> more exploitation
    
    # Normalize sigma to get a sense of where we are
    # Use the ratio of mean sigma to max sigma as a proxy
    if sigma_max > 1e-9:
        sigma_ratio = sigma_mean / sigma_max
    else:
        sigma_ratio = 1.0
    
    # kappa should be higher when sigma_ratio is high (early on)
    # Base kappa in [1, 4], scaled by sigma_ratio
    base_kappa = 2.5
    # Dynamic adjustment: increase kappa when uncertainty is high
    kappa = base_kappa * (0.5 + 0.5 * sigma_ratio)
    
    # Compute UCB scores: mu - kappa * sigma (since we're minimizing)
    scores = mu - kappa * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
