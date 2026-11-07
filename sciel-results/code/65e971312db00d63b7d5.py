"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for minimizing Branin 2D via Bayesian optimization.
    Uses a hybrid strategy: primarily Expected Improvement (EI) for exploitation,
    but with a bounded exploration component for regions with high uncertainty.
    
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviation predictions
    f_best: scalar, best observed value so far
    Returns: (2000,) array of scores, higher is better
    """
    # Handle edge case: sigma might be zero or very small
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is always non-negative. Higher EI means better expected improvement.
    # However, EI can be very small for points that are far from the optimum
    # but have high uncertainty. To encourage exploration of unexplored regions,
    # we add a small exploration bonus proportional to sigma, but bounded.
    
    # Scale EI to a reasonable range and add a small sigma-based bonus
    # The bonus is capped to prevent it from dominating the EI term
    # We use a fixed small coefficient for the exploration bonus
    # Since EI is in objective units, we need to be careful with scaling
    
    # Alternative: Use a pure EI approach, which is known to work well
    # for BO with a small number of evaluations. The seed program used EI
    # and achieved a score of 1.0 (the best). Let's stick with EI.
    
    # However, to make it more robust, let's consider a hybrid approach
    # where we blend EI with a small UCB-like term.
    
    # Actually, the inspiration says "Replace the heuristic hybrid EI+UCB with
    # a pure Gaussian Expected Improvement score". And the parent program
    # (score 1.0) used pure EI. The current program (score 0.0159) uses UCB.
    # So we should go back to pure EI.
    
    # But let's make sure the EI computation is correct and robust.
    # For large negative z (mu >> f_best), EI should be close to 0.
    # For large positive z (mu << f_best), EI should be close to (f_best - mu).
    
    # The standard formula handles this well.
    scores = ei
    
    # Ensure all scores are finite and non-negative
    # EI can be very small but non-negative. If numerical issues cause
    # negative values, clamp to 0.
    scores = np.maximum(scores, 0.0)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
