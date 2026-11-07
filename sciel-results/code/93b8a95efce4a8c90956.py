"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: (2000,) array of GP mean predictions in objective units
        sigma: (2000,) array of GP standard deviations in objective units
        f_best: float, lowest observed value
        cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    # Deadline guard: track elapsed time
    t_start = time.time()
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        return np.zeros_like(mu)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Use scipy.stats.norm for CDF and PDF
    from scipy.stats import norm
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Compute Expected Improvement per unit cost (EIPC)
    cost_safe = np.maximum(cost, eps)
    eipc = ei / cost_safe
    
    # Dynamic threshold: reject candidates with low EIPC
    # As f_best improves (gets lower), we become more selective
    # The progress factor increases as f_best approaches the minimum (0)
    # Since the minimum is 0, progress = 1 - f_best (when f_best > 0)
    # But we need to be careful: f_best starts high and decreases toward 0
    
    # Use a penalty that increases with progress
    # When f_best is large (early), we tolerate higher cost
    # When f_best is small (late), we penalize cost heavily
    
    # Progress metric: how close are we to the minimum?
    # The minimum is 0, so progress = max(0, 1 - f_best)
    # But f_best can be negative in some cases (though minimum is 0)
    # Let's use a normalized progress based on typical f_best values
    
    # For powell 4d on [-4,5]^4, initial f_best is typically O(1-10)
    # As we optimize, f_best decreases toward 0
    
    # Use a smooth penalty factor that increases as f_best decreases
    # penalty = 1 + alpha * progress
    # where progress is in [0, 1]
    
    # Estimate progress: assume f_best starts around 5 and goes to 0
    # progress = clip(1 - f_best / 5, 0, 1)
    # But this is heuristic. Let's use a more robust approach.
    
    # The cost is 1 + mean(x_scaled), where x_scaled is in [0,1]
    # So cost is in [1, 2]
    # We want to prefer cheaper points when we're making progress
    
    # Use a multiplicative penalty that grows with progress
    # Let's define progress based on how many evaluations we've done
    # But we don't have that info directly. We can infer from f_best.
    
    # Alternative: use the distribution of costs to set a threshold
    # Reject candidates whose EIPC is below a quantile
    
    # Compute a dynamic threshold based on the current EIPC distribution
    # Use the 50th percentile as a baseline, and adjust based on progress
    
    # For now, use a simple approach: penalize cost more as f_best improves
    # penalty_factor = 1 + max(0, 1 - f_best) * 2
    # This gives penalty_factor in [1, 3] for f_best in [0, 1]
    
    # But f_best can be larger initially. Let's normalize differently.
    # Use: progress = 1 / (1 + f_best)
    # This gives progress in (0, 1], with 1 when f_best=0 and approaching 0 as f_best grows
    
    progress = 1.0 / (1.0 + max(0.0, f_best))
    
    # Penalty factor: increases from 1 to 2 as progress goes from 0 to 1
    penalty_factor = 1.0 + progress
    
    # Apply penalty to EIPC
    scores = eipc / penalty_factor
    
    # Additionally, add a small exploration bonus for points with high uncertainty
    # that are in cheap regions, to avoid getting stuck in local optima
    # But be careful not to over-explore
    
    # Return finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure all scores are finite
    if not np.all(np.isfinite(scores)):
        scores = np.zeros_like(mu)
    
    return scores
# EVOLVE-BLOCK-END
