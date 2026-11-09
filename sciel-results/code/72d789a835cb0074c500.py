"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Deterministic Worst-Case Improvement (WCI) per unit cost.
    
    WCI = max(0, f_best - mu - k * sigma)
    Score = WCI / cost
    
    This shifts from probabilistic expected improvement to a robust,
    minimax-regret style score that penalizes high uncertainty directly.
    """
    # Exploration constant: balances exploitation vs exploration
    # Increased to 2.0 to encourage more exploration in high-uncertainty regions
    # which helps escape local minima in the multimodal Shekel 4d landscape.
    k = 2.0
    
    # Compute worst-case improvement (deterministic, no CDF/PDF)
    # f_best is the current best (lowest) value.
    # mu is the GP mean prediction.
    # sigma is the GP standard deviation.
    # We want high scores for points where (f_best - mu) is large (mu << f_best)
    # and where uncertainty (sigma) is not too high (robustness).
    wci = np.maximum(0.0, f_best - mu - k * sigma)
    
    # Guard against zero or negative costs
    cost_safe = np.maximum(cost, 1e-9)
    
    # Score: improvement per unit cost (higher = evaluate next)
    scores = wci / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
