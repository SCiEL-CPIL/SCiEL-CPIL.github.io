import numpy as np
import time
from scipy.stats import norm

# Global timer for safety
_start_time = time.time()
_DEADLINE = 320.0  # 10s margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses a hybrid EI + UCB approach with adaptive blending.
    Early iterations favor exploration (higher UCB weight),
    later iterations favor exploitation (higher EI weight).
    
    Scores are normalized to [0, 1] to ensure comparability between
    EI (expected improvement) and UCB (upper confidence bound).
    """
    # Check time budget
    if time.time() - _start_time > _DEADLINE:
        return np.zeros_like(mu)

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-10)
    
    # 1. Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Clip z to avoid overflow in exp for phi, though Phi is safe
    z_clipped = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative (numerical errors can cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # 2. Calculate UCB (Upper Confidence Bound) for minimization
    # Score = -mu + beta * sigma
    # We use a fixed beta for the UCB component, as the blending handles adaptation.
    # A moderate beta like 2.0 is standard.
    beta_ucb = 2.0
    ucb = -mu + beta_ucb * sigma
    
    # 3. Normalize both scores to [0, 1]
    # This makes them comparable regardless of the scale of mu/sigma
    def normalize(scores):
        min_s = np.min(scores)
        max_s = np.max(scores)
        range_s = max_s - min_s
        if range_s < 1e-10:
            return np.zeros_like(scores)
        return (scores - min_s) / range_s

    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb)
    
    # 4. Adaptive Blending
    # f_best is the current minimum. 
    # High f_best -> Early stage -> More Exploration (UCB) -> Lower alpha (weight on EI)
    # Low f_best  -> Late stage  -> More Exploitation (EI)  -> Higher alpha (weight on EI)
    
    # Clip f_best to [0, 1] to be safe
    fb = np.clip(f_best, 0.0, 1.0)
    
    # Alpha is the weight for EI.
    # When fb is high (e.g., 0.5), alpha is low (e.g., 0.2) -> Mostly UCB
    # When fb is low (e.g., 0.0), alpha is high (e.g., 1.0) -> Mostly EI
    # Linear mapping: alpha = 1.0 - fb
    # To be more aggressive in exploitation early on if needed, we can adjust.
    # But 1-fb is a good start.
    alpha = 1.0 - fb
    
    # Final Score
    score = alpha * ei_norm + (1.0 - alpha) * ucb_norm
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
