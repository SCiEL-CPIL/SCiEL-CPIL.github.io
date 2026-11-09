"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of styblinski tang 1d.
    
    Uses Quantile-based Expected Improvement (QEI) targeting the 95th percentile
    of the posterior predictive distribution, combined with a small exploration
    bonus to handle multimodal landscapes.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array_like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate. Higher is better.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Deadline guard: if we've been running too long, return a safe fallback
    deadline = 300.0  # 300 seconds total, leave margin
    elapsed = time.time() - start_time
    
    if elapsed > deadline - 5:
        # Fallback: return simple EI to be safe
        z = (f_best - mu) / sigma
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        ei = (f_best - mu) * cdf_z + sigma * pdf_z
        scores = np.where(np.isfinite(ei), ei, 0.0)
        return scores
    
    # Quantile-based Expected Improvement (QEI)
    # Target the 95th percentile: z_0.95 = 1.6449
    # QEI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu + k*sigma) / sigma
    # This effectively shifts the improvement threshold by k*sigma to target higher confidence
    
    # Use a moderate quantile shift (k=1.0 corresponds to ~84th percentile, k=1.64 to 95th)
    # We'll use a blend: primary QEI at 95th percentile with a small UCB-like exploration term
    
    k_quantile = 1.0  # Moderate quantile shift for robustness in multimodal landscape
    
    # Compute z-score for QEI
    z_qei = (f_best - mu + k_quantile * sigma) / sigma
    
    cdf_z = norm.cdf(z_qei)
    pdf_z = norm.pdf(z_qei)
    
    # QEI: Expected improvement at the shifted quantile
    qei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Add a small exploration bonus to handle multimodality
    # The exploration term encourages visiting high-uncertainty regions
    # Scale exploration by the current best to normalize
    exploration_weight = 0.1
    exploration = exploration_weight * sigma
    
    scores = qei + exploration
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline before returning
    elapsed = time.time() - start_time
    if elapsed > deadline - 1:
        # Return a simplified version if we're close to deadline
        z = (f_best - mu) / sigma
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        ei = (f_best - mu) * cdf_z + sigma * pdf_z
        scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
