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
    
    # Estimate the current level of uncertainty to dynamically adjust kappa.
    # Early in the optimization, sigma is high everywhere, so we want high exploration.
    # Later, sigma is low in explored regions, so we want more exploitation.
    # We use the maximum sigma as a proxy for the remaining uncertainty.
    sigma_max = np.max(sigma)
    
    # Clamp sigma_max to a reasonable range to avoid extreme kappa values.
    # Standardized GP outputs usually have sigma in [0, 1] or slightly above.
    sigma_norm = np.clip(sigma_max, 0.1, 1.0)
    
    # Base kappa of 2.5 is a common starting point for 3D problems.
    # Scaling by sigma_norm reduces kappa as uncertainty decreases.
    kappa = 2.5 * sigma_norm
    
    # Compute UCB scores: mu - kappa * sigma (since we're minimizing)
    scores = mu - kappa * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
