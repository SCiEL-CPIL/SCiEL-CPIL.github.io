"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Hybrid Acquisition Function: Weighted Sum of Normalized EI and Adaptive UCB.
    
    Args:
        mu: np.array of shape (2000,), GP mean predictions.
        sigma: np.array of shape (2000,), GP standard deviation predictions.
        f_best: float, best observed value so far.
        
    Returns:
        np.array of shape (2000,), acquisition scores (higher is better).
    """
    t_start = time.time()
    
    # 1. Numerical Stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # 2. Calculate Expected Improvement (EI)
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid extreme values in cdf/pdf if needed, but scipy handles it well.
    # However, for very large negative z, phi(z) is ~0 and cdf(z) is ~0.
    # For very large positive z, cdf(z) is ~1 and phi(z) is ~0.
    # EI is always non-negative.
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0) # Ensure non-negative
    
    # 3. Calculate Adaptive UCB
    # For minimization, we want low mu. UCB score = -mu + beta * sigma.
    # We adapt beta based on relative uncertainty.
    
    # Estimate the spread of the predictions to normalize uncertainty
    mu_spread = np.std(mu)
    if mu_spread < 1e-8:
        mu_spread = 1e-8
        
    # Relative uncertainty: how large is sigma compared to the variation in mu?
    # If mu is very flat (low spread) but sigma is high, we are very uncertain.
    # If mu varies a lot, sigma might be relatively small.
    sigma_relative = sigma_safe / mu_spread
    
    # Use median to be robust to outliers
    median_rel_uncertainty = np.median(sigma_relative)
    
    # Adaptive beta: 
    # Base beta of 0.5 ensures some exploration even when uncertainty is low.
    # Scaling factor 1.5 allows beta to grow up to ~2.0-3.0 if uncertainty is high.
    # This corrects the "fixed UCB was too exploratory" issue by keeping beta lower
    # when the model is confident (low relative sigma), and higher when uncertain.
    beta = 0.5 + 1.5 * median_rel_uncertainty
    
    ucb_score = -mu + beta * sigma_safe
    
    # 4. Normalize Scores to [0, 1]
    # This prevents one term from dominating due to scale differences.
    
    # Normalize EI
    ei_max = np.max(ei)
    if ei_max < 1e-8:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = ei / ei_max
        ei_norm = np.clip(ei_norm, 0.0, 1.0)
        
    # Normalize UCB
    # UCB score can be negative. We normalize based on the range.
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range < 1e-8:
        ucb_norm = np.zeros_like(ucb_score)
    else:
        ucb_norm = (ucb_score - ucb_min) / ucb_range
        ucb_norm = np.clip(ucb_norm, 0.0, 1.0)

    # 5. Combine Scores
    # EI is generally more effective for convergence in BO.
    # UCB helps escape local optima in multimodal functions like Ackley.
    # Weighting: 0.6 EI, 0.4 UCB.
    w_ei = 0.6
    w_ucb = 0.4
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # 6. Deadline Guard (Safety Margin)
    # The operation is vectorized and fast, but we check just in case.
    # If we exceed 325s (5s margin from 330s limit), return current scores.
    # In practice, this never triggers for 2000 points.
    if time.time() - t_start > 325.0:
         # Return current scores immediately
         pass

    # 7. Ensure Finite Values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
