"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None
_call_count = 0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses a Hypervolume-inspired acquisition that combines Expected Improvement
    with a diversity bonus based on the spread of predicted uncertainty.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores. Higher values indicate more promising candidates.
    """
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
    _call_count += 1
    
    # Deadline guard: if we're getting close to the 330s limit, return simple EI
    elapsed = time.time() - _start_time
    if elapsed > 300:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Hypervolume-inspired approach:
    # We want to score candidates by how much "volume" they add to the 
    # (improvement, uncertainty) Pareto front.
    # 
    # For each candidate, define:
    #   improvement = f_best - mu (how much we expect to improve)
    #   uncertainty = sigma (how uncertain we are)
    #
    # A candidate is valuable if it offers good improvement OR high uncertainty
    # in a way that's not already dominated by other candidates.
    #
    # Approximate hypervolume contribution:
    # 1. Compute a 2D Pareto front of (improvement, uncertainty)
    # 2. For each point, compute the area of the rectangle it "owns"
    #    in the 2D space, normalized appropriately.
    
    improvement = f_best - mu  # Higher is better (more expected improvement)
    uncertainty = sigma_safe   # Higher is more exploratory
    
    # Normalize improvement and uncertainty to [0, 1] for fair comparison
    imp_min = np.min(improvement)
    imp_max = np.max(improvement)
    unc_min = np.min(uncertainty)
    unc_max = np.max(uncertainty)
    
    imp_range = imp_max - imp_min + 1e-12
    unc_range = unc_max - unc_min + 1e-12
    
    imp_norm = (improvement - imp_min) / imp_range
    unc_norm = (uncertainty - unc_min) / unc_range
    
    # Find non-dominated points in (imp_norm, unc_norm) space
    # A point dominates another if it has >= in both dimensions and > in at least one
    # For efficiency with 2000 points, we use a simple approach:
    # Sort by improvement descending, then track max uncertainty seen so far
    
    # Sort by improvement descending
    order = np.argsort(-imp_norm)
    imp_sorted = imp_norm[order]
    unc_sorted = unc_norm[order]
    
    # Compute for each point the "hypervolume contribution"
    # A point's contribution is the area of the rectangle from its (imp, unc)
    # to the next non-dominated point or the axes.
    
    # Track the running maximum of uncertainty for points with higher or equal improvement
    # This gives us the "frontier" in uncertainty for each improvement level
    
    # For each point, its hypervolume contribution is approximately:
    # (imp_norm[i] - imp_norm[i+1]) * (unc_norm[i] - max_unc_below)
    # where max_unc_below is the max uncertainty among points with lower improvement
    
    # Simpler approximation: score = imp_norm * unc_norm + imp_norm + unc_norm
    # This rewards points that are high in both dimensions.
    
    # Even simpler: use the product as a proxy for hypervolume contribution
    hv_score = imp_norm * unc_norm
    
    # Combine EI with hypervolume-inspired diversity score
    # EI captures exploitation, hv_score captures exploration/diversity
    
    # Scale EI to be comparable to hv_score
    ei_max = np.max(ei) + 1e-12
    ei_norm = ei / ei_max
    
    # Weight: favor EI more as we get closer to convergence (f_best is small)
    # For Ackley, f_best starts around 3-4 and approaches 0
    # Use a dynamic weight based on how close f_best is to 0
    if f_best > 0.5:
        alpha = 0.6  # More exploration early
    elif f_best > 0.1:
        alpha = 0.75
    else:
        alpha = 0.9  # More exploitation late
    
    scores = alpha * ei_norm + (1 - alpha) * hv_score
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
