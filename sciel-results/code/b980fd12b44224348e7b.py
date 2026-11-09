"""Rewritten program for Bayesian optimization of Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores using a Quantile-based Expected Improvement
    approach that combines:
    1. Probability of improvement by a margin (robust, bounded score)
    2. Lower Confidence Bound (LCB) for balanced exploration-exploitation
    
    For minimization:
    - Q-EI: P(f(x) < f_best - epsilon) = Phi((f_best - epsilon - mu) / sigma)
      where epsilon = 0.1 * |f_best| (adaptive margin)
    - LCB: -mu + kappa * sigma (kappa adapts to f_best)
    
    This creates a robust, bounded score that strongly encourages finding
    points significantly better than f_best while maintaining exploration.
    """
    # Deadline guard: 330s limit. Return immediately if close to limit.
    if time.time() - _START_TIME > 320:
        return np.zeros_like(mu)
    
    # Clip sigma to avoid division by zero and ensure positive values
    sigma = np.maximum(sigma, 1e-10)
    
    # Compute z-scores
    # For Q-EI: probability of improving by margin epsilon
    epsilon = 0.1 * abs(f_best)
    z_qei = (f_best - epsilon - mu) / sigma
    
    # For LCB: standard z-score
    z_lcb = (f_best - mu) / sigma
    
    # Standard normal CDF using scipy
    try:
        from scipy.stats import norm
        Phi_qei = norm.cdf(z_qei)
        Phi_lcb = norm.cdf(z_lcb)
    except ImportError:
        sqrt2 = np.sqrt(2.0)
        Phi_qei = 0.5 * (1.0 + np.erf(z_qei / sqrt2))
        Phi_lcb = 0.5 * (1.0 + np.erf(z_lcb / sqrt2))
    
    # Q-EI score: bounded in [0, 1], high when improvement is likely
    qei_score = Phi_qei
    
    # LCB score: -mu + kappa*sigma, normalized to be comparable
    # kappa adapts: higher when f_best is far from 0 (more exploration)
    kappa = 1.5 + 0.5 * np.clip(f_best / 5.0, 0.0, 1.0)
    lcb_raw = -mu + kappa * sigma
    
    # Normalize LCB to [0, 1] range using min-max within the candidate set
    lcb_min = np.min(lcb_raw)
    lcb_max = np.max(lcb_raw)
    lcb_range = lcb_max - lcb_min
    if lcb_range > 1e-10:
        lcb_score = (lcb_raw - lcb_min) / lcb_range
    else:
        lcb_score = np.ones_like(lcb_raw) * 0.5
    
    # Combine Q-EI and LCB scores
    # Weight Q-EI more heavily as f_best approaches 0
    w_qei = 0.6 + 0.4 * np.clip(1.0 - f_best / 5.0, 0.0, 1.0)
    w_lcb = 1.0 - w_qei
    
    score = w_qei * qei_score + w_lcb * lcb_score
    
    # Ensure scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
