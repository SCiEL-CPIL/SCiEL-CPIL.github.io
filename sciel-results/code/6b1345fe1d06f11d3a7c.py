import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) and
    Upper Confidence Bound (UCB) with robust z-score normalization and clipping.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB term for exploration
    beta = 1.0
    ucb = -mu + beta * sigma_safe
    
    # Robust Z-score normalization with clipping
    def _robust_zscore(arr):
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-12:
            return np.zeros_like(arr)
        z_scores = (arr - mean) / std
        # Clip to avoid outlier dominance
        return np.clip(z_scores, -3.0, 3.0)

    ei_norm = _robust_zscore(ei)
    ucb_norm = _robust_zscore(ucb)
    
    # Weights: 70% EI, 30% UCB
    w_ei = 0.7
    w_ucb = 0.3
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
