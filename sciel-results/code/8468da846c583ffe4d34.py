"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard (though acquisition is fast)
_start_time = time.time()
_DEADLINE = 300.0  # Safety margin from 330s kill

def acquisition(mu, sigma, f_best):
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a valid array of zeros or something finite to avoid crash
        # But the evaluator kills us anyway. Just return fast.
        return np.zeros_like(mu)

    # Handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate standard EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Structural Change: Add an exploration term based on sigma
    # This helps escape local minima and explores uncertain regions
    # Weight 0.5 is a heuristic balance
    exploration_term = 0.5 * sigma_safe
    
    # Alternative: Adaptive EI with xi
    # Let's try the hint's approach: Scale-adaptive EI
    # Estimate scale from the distribution of mu
    # Use IQR of mu to determine xi
    q75, q25 = np.percentile(mu, [75, 25])
    iqr_mu = q75 - q25
    if iqr_mu < 1e-6:
        xi = 0.0
    else:
        # Use a fraction of the IQR as the noise parameter
        xi = 0.1 * iqr_mu
        
    z_adaptive = (f_best - xi - mu) / sigma_safe
    ei_adaptive = (f_best - xi - mu) * norm.cdf(z_adaptive) + sigma_safe * norm.pdf(z_adaptive)
    
    # Combine Adaptive EI and Exploration
    # If we are early on, xi is small, EI_adaptive ~ EI.
    # If we are late, xi might be larger?
    # Actually, if we are late, mu is concentrated, IQR is small, xi is small.
    # So Adaptive EI is similar to EI.
    
    # Let's stick to the simpler and more robust "EI + Exploration"
    # score = ei + exploration_term
    
    # Or just use the Adaptive EI?
    # The hint said it improved score.
    # Let's use a mix: 0.5 * ei + 0.5 * ei_adaptive + 0.1 * sigma?
    # No, keep it simple.
    
    # I will use EI + 0.5 * sigma.
    # This is a standard "Boosted EI" or "UCB-like" modification.
    
    scores = ei + 0.5 * sigma_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
