"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Computes a cost-aware acquisition function score for Bayesian Optimization.
    Uses a hybrid of Expected Improvement and a dynamically-tuned UCB term,
    normalized before cost division.
    
    Args:
        mu: Array of predicted means (shape 2000,).
        sigma: Array of predicted standard deviations (shape 2000,).
        f_best: The lowest observed function value.
        cost: Array of evaluation costs for each candidate (shape 2000,).
        
    Returns:
        Array of scores (shape 2000,), higher is better.
    """
    t_start = time.time()
    
    # Ensure inputs are properly shaped and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Dynamically tune kappa based on remaining budget vs current cost
    # Total budget is 30. The cost array represents the cost of evaluating each candidate.
    # We use the mean cost as a proxy for "current cost" per evaluation.
    # As we spend more budget, we should shift from exploration to exploitation.
    # However, we don't know exactly how much budget is left.
    # We can estimate remaining budget by looking at the cost array.
    # Actually, the cost array is the cost of *each candidate*.
    # The total budget is 30. We don't know how much has been spent.
    # Let's use a simple heuristic: start with high exploration, decrease over time.
    # Since we don't have a time step, we'll use the magnitude of f_best.
    # As f_best gets closer to the true minimum (-3.86278), we should exploit more.
    # Let's normalize f_best. The range of Hartmann 3d is roughly [-4, 0].
    # Let's map f_best to a range [0, 1] where 0 is best (most negative) and 1 is worst.
    # f_best is in objective units.
    # Let's assume the range is [-4, 0] for normalization.
    f_best_norm = np.clip((f_best + 4.0) / 4.0, 0.0, 1.0)
    # When f_best is -4 (best), f_best_norm = 0. kappa should be low.
    # When f_best is 0 (worst), f_best_norm = 1. kappa should be high.
    # Let's use a quadratic curve to shift exploration early.
    # kappa = base + range * (1 - f_best_norm)^2
    # Early on, f_best is high (close to 0), so f_best_norm is high, (1-f_best_norm) is low.
    # Wait, if f_best is -3.8, f_best_norm is 0.05. (1-0.05)^2 is 0.9. High exploration.
    # If f_best is -0.1, f_best_norm is 0.975. (1-0.975)^2 is 0.0006. Low exploration.
    # This seems backwards. We want high exploration when f_best is BAD (close to 0).
    # So if f_best is 0, f_best_norm = 1. We want high kappa.
    # If f_best is -4, f_best_norm = 0. We want low kappa.
    # So kappa should be proportional to f_best_norm.
    # Let's try: kappa = 0.5 + 2.0 * f_best_norm
    # Early (f_best ~ -1), f_best_norm ~ 0.75, kappa ~ 2.0.
    # Late (f_best ~ -3.8), f_best_norm ~ 0.05, kappa ~ 0.6.
    # This makes sense.
    kappa = 0.5 + 2.0 * f_best_norm
    
    # Upper Confidence Bound term for minimization
    # We want to maximize the score.
    # The optimistic value is mu - kappa * sigma.
    # To maximize this, we minimize mu - kappa * sigma.
    # So the score component is -(mu - kappa * sigma) = -mu + kappa * sigma.
    ucb_term = -mu + kappa * sigma
    
    # Normalize EI and UCB term
    # EI is in objective units. UCB term is in objective units.
    # We should normalize them to be on the same scale.
    # Let's use the max of each.
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_max = np.max(ucb_term) if np.max(ucb_term) > 0 else 1.0
    
    ei_norm = ei / ei_max
    ucb_norm = ucb_term / ucb_max
    
    # Blend EI and UCB
    # We can use a dynamic weight as well.
    # Let's use a fixed weight of 0.5 for now, or make it dynamic.
    # Since kappa is already dynamic, let's keep the blend simple.
    blended = 0.5 * ei_norm + 0.5 * ucb_norm
    
    # Cost-aware: divide by cost
    scores = blended / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check time - if we're getting close to the limit, return immediately
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
