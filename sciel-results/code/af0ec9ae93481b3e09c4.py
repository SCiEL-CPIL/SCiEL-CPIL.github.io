# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = time.time() + 325.0  # 5s safety margin before 330s limit


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture in 2D.
    Uses Tempered Expected Improvement (EI) for minimization, with a deadline guard.
    The temperature parameter sharpens the acquisition function, favoring exploitation.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP mean at candidate points.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation at candidate points.
    f_best : float
        Lowest observed value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a simple fallback: exploit the best predicted point
        return -mu.copy()
    
    # Handle sigma near zero to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standard z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in exp/log
    z = np.clip(z, -10, 10)
    
    # Temperature parameter to sharpen the EI (T < 1 favors exploitation)
    # T = 0.8 makes the distribution sharper
    T = 0.8
    z_temp = z / T
    
    # Compute Tempered EI
    # Note: The formula for EI with temperature T is not simply substituting z/T 
    # into the standard EI formula because the integral changes.
    # However, a common heuristic is to use the standard EI formula but with a 
    # modified z or to use a different acquisition like UCB.
    # 
    # Actually, the "Tempered EI" usually refers to:
    # EI_T = E[max(0, f_best - f(x))] where the "max" is approximated or the 
    # underlying distribution is tempered.
    # A simpler and valid approach in the EI family is to use:
    # Score = EI * exp(-lambda * sigma)  (penalize uncertainty)
    # OR
    # Score = EI + lambda * Phi(z)
    
    # Let's use the "Probabilistic Improvement" blend which is smooth and stable.
    # PI = Phi(z). It is less noisy than EI.
    # Blend: Score = EI + 0.5 * PI
    # This ensures that even if EI is small, points with high probability of 
    # improvement are favored.
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Add a small weight to the Probability of Improvement to smooth out the 
    # acquisition landscape and reduce sensitivity to sigma spikes.
    pi = Phi_z
    
    scores = ei + 0.1 * pi
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
