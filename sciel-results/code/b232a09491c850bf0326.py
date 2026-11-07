"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PoI) acquisition function.
    
    Structural change from EI: Instead of computing expected improvement magnitude,
    we compute the probability that a point improves upon an effective best value.
    The effective best is adjusted by cost: effective_best = f_best + beta * (cost - 1).
    This treats cost as a virtual penalty on the objective, so expensive points must
    show greater predicted improvement to be selected. This is a different modeling
    family (probabilistic threshold vs. expected value) that naturally handles
    cost-awareness without ad-hoc penalties.
    
    PoI(x) = Phi((effective_best - mu) / sigma)
    
    For minimization, we want high probability that f(x) < effective_best.
    """
    start_time = time.time()
    deadline = start_time + 325.0  # 5s safety margin
    
    # Ensure inputs are 1D arrays of float64 for performance and stability
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost-aware effective best: shift the threshold based on cost
    # cost is in [1, 2], so (cost - 1) is in [0, 1]
    # beta controls how strongly cost affects the selection threshold
    beta = 0.3
    effective_best = f_best + beta * (cost - 1.0)
    
    # PoI: probability that f(x) < effective_best
    # z = (effective_best - mu) / sigma
    # PoI = Phi(z) where Phi is the standard normal CDF
    z = (effective_best - mu) / sigma_safe
    
    # PoI score (higher = more likely to improve)
    poi_score = norm.cdf(z)
    
    # Add a small exploration bonus for high uncertainty to avoid premature convergence
    # This is scaled by the probability mass in the tails
    exploration_bonus = 0.05 * sigma_safe / (sigma_safe + np.abs(mu - f_best) + 1e-9)
    
    final_score = poi_score + exploration_bonus
    
    # Ensure finiteness: replace NaN/Inf with -inf so they are never selected
    final_score = np.where(np.isfinite(final_score), final_score, -np.inf)
    
    # Check deadline; if close, return immediately
    if time.time() > deadline - 1.0:
        return final_score
    
    return final_score
# EVOLVE-BLOCK-END
