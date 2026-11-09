"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    Uses a hybrid of Expected Improvement (EI) and Maximum Entropy (Log-Sigma).
    The blend factor is adaptive based on the current best value (f_best),
    shifting from exploration (Entropy) to exploitation (EI) as f_best improves.
    """
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    
    # 1. Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    # Clip z for numerical stability
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute Maximum Entropy proxy: log(sigma)
    # High sigma -> High uncertainty -> High entropy
    # We use log to prevent sigma from dominating if it's very large
    entropy = np.log(sigma)
    
    # 3. Normalize scores to [0, 1] for blending
    def normalize(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-10:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    ei_norm = normalize(ei)
    entropy_norm = normalize(entropy)
    
    # 4. Adaptive Blending
    # Progress metric: 0.0 when f_best is bad (high), 1.0 when f_best is good (low)
    # Branin min is ~0.4. Initial f_best can be ~30.
    # Let's define progress such that:
    # f_best >= 10.0 -> progress = 0.0 (Full Exploration)
    # f_best <= 1.0  -> progress = 1.0 (Full Exploitation)
    # Linear interpolation in between.
    
    f_high = 10.0
    f_low = 1.0
    
    if f_best >= f_high:
        progress = 0.0
    elif f_best <= f_low:
        progress = 1.0
    else:
        progress = (f_high - f_best) / (f_high - f_low)
    
    # Clamp progress
    progress = np.clip(progress, 0.0, 1.0)
    
    # Blend weight for EI. 
    # We want to start with more Entropy (exploration) and move to EI (exploitation).
    # So weight_EI = progress, weight_Entropy = 1 - progress.
    # However, pure entropy exploration can be inefficient even early on.
    # Let's keep a minimum exploitation weight of 0.2.
    w_ei = 0.2 + 0.8 * progress
    w_ent = 1.0 - w_ei
    
    # 5. Combined Score
    scores = w_ei * ei_norm + w_ent * entropy_norm
    
    # Scale up to ensure discrimination
    scores = scores * 10.0
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
