"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Uses Expected Improvement (EI) with a cost-aware penalty that balances
    exploration against evaluation cost. The key insight is that we want to
    maximize improvement per unit cost, but we also need to handle edge cases
    where sigma is very small or cost is very small.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from the GP surrogate
    sigma : array-like, shape (2000,)
        Standard deviation predictions from the GP surrogate
    f_best : float
        Best (lowest) observed function value
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Set a deadline guard - we have 300s per evaluation, use 280s as safety margin
    deadline = time.time() + 280.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    scores = np.zeros(n, dtype=np.float64)
    
    # Handle edge cases
    # Avoid division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score for expected improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware score: EI / cost
    # But we need to be careful: if cost is very small, this could blow up
    # Also, if ei is negative (mu > f_best), we don't want to evaluate
    cost_safe = np.maximum(cost, 1e-10)
    
    # Base score: EI per unit cost
    scores = ei / cost_safe
    
    # Penalize points where the GP is very uncertain (high sigma) relative to 
    # the potential improvement - this encourages exploitation when we're close
    # to the optimum but still allows exploration when uncertainty is high
    
    # Add a small exploration bonus for high uncertainty regions, but weight it
    # by how much improvement is possible
    # This helps avoid getting stuck in local optima
    
    # Normalize sigma for the exploration bonus
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    sigma_norm = sigma_safe / (sigma_max + 1e-10)
    
    # Exploration bonus: small weight, encourages visiting uncertain regions
    # but only if there's potential for improvement
    exploration_weight = 0.1
    scores += exploration_weight * sigma_norm * norm.pdf(z)
    
    # Ensure scores are finite
    scores[~np.isfinite(scores)] = 0.0
    
    # Check deadline - if we're close to the limit, return immediately
    if time.time() > deadline - 5.0:
        return scores
    
    # Additional refinement: use a hybrid approach
    # For points with high EI, trust the EI more
    # For points with low EI but high uncertainty, allow more exploration
    
    # Compute a confidence-weighted score
    # Higher confidence (lower sigma relative to improvement) gets more weight
    confidence = np.exp(-sigma_safe / (np.std(mu) + 1e-10))
    
    # Blend between pure EI/cost and a more exploratory score
    blend = 0.8
    scores_final = blend * (ei / cost_safe) + (1 - blend) * (ei / cost_safe + 0.2 * sigma_norm * norm.pdf(z))
    
    # Ensure finite
    scores_final[~np.isfinite(scores_final)] = 0.0
    
    # If we have time, do a second pass with adaptive weights
    if time.time() < deadline - 10.0:
        # Adaptive: if we've found good points, be more exploitative
        # Otherwise, be more exploratory
        if f_best < -3.5:
            # Close to optimum, exploit more
            explo_weight = 0.9
        elif f_best < -3.0:
            explo_weight = 0.7
        else:
            # Early on, explore more
            explo_weight = 0.5
        
        scores_adaptive = explo_weight * (ei / cost_safe) + (1 - explo_weight) * (0.5 * ei / cost_safe + 0.5 * sigma_norm * norm.pdf(z) / cost_safe)
        scores_adaptive[~np.isfinite(scores_adaptive)] = 0.0
        
        # Use the adaptive version if it looks reasonable
        if np.std(scores_adaptive) > 1e-10:
            return scores_adaptive
    
    return scores
# EVOLVE-BLOCK-END
