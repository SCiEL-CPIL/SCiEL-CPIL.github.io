import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 320.0  # safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Quantile-based Expected Improvement (QEI) for minimization.
    
    Instead of standard EI (which maximizes mean improvement), QEI targets
    the alpha-quantile of the improvement distribution, providing a more
    robust exploitation signal in multimodal landscapes.
    
    Args:
        mu: numpy array of shape (N,) posterior mean.
        sigma: numpy array of shape (N,) posterior standard deviation.
        f_best: float, the best observed function value so far.
        
    Returns:
        numpy array of shape (N,) scores. Higher scores indicate better candidates.
    """
    # Check time budget
    if time.time() > _DEADLINE:
        # Fallback: return scores based on low mu (exploitation)
        return -mu.copy()
    
    # Avoid division by zero
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # z-score for standard EI
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Standard EI for minimization
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Quantile-based approach: target the 95th percentile of improvement
    # The improvement distribution is: I = max(f_best - f(x), 0)
    # where f(x) ~ N(mu, sigma^2)
    # 
    # The 95th percentile of f(x) is mu + 1.645*sigma
    # So the 95th percentile of improvement (lower bound on f) is:
    # f_05 = mu - 1.645*sigma  (5th percentile of f, i.e., 95% confident f > f_05)
    # QEI = max(f_best - f_05, 0) = max(f_best - mu + 1.645*sigma, 0)
    
    alpha = 1.645  # 95th percentile z-value
    
    # Quantile-based improvement: optimistic lower bound on improvement
    qei = np.maximum(f_best - mu + alpha * sigma_safe, 0.0)
    
    # Blend: use QEI as primary signal, with small EI component for smoothness
    # Normalize both to [0, 1] range for blending
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    qei_max = np.max(qei) if np.max(qei) > 0 else 1.0
    
    ei_norm = ei / ei_max
    qei_norm = qei / qei_max
    
    # Weighting: favor QEI for robustness, keep some EI for gradient information
    # Early on (high uncertainty), QEI helps explore; later, EI refines
    weights_qei = 0.7
    weights_ei = 0.3
    
    scores = weights_qei * qei_norm + weights_ei * ei_norm
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
