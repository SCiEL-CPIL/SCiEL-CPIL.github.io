"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # For minimization, EI is the expected amount by which the new point
    # improves upon the current best f_best.
    # E[f(x) < f_best] * (f_best - E[f(x) | f(x) < f_best])
    # This simplifies to: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware multiplicative gate: EI * exp(-k * cost)
    # This exponentially suppresses high-cost points unless EI is exceptionally large.
    # Calibrate k to the expected EI scale.
    # For cosine8, f ranges roughly [-0.8, 1], so f_best starts near 0-1 and improves toward -0.8.
    # Typical EI magnitudes are O(0.1-1). Cost is in [1, 2] (1 + mean(x scaled to [0,1])).
    # We want exp(-k * cost) to be ~0.5-0.9 for low cost (~1) and ~0.1-0.3 for high cost (~2).
    # If k=1.5: exp(-1.5*1)=0.22, exp(-1.5*2)=0.05 -> too aggressive
    # If k=0.5: exp(-0.5*1)=0.61, exp(-0.5*2)=0.37 -> reasonable
    # Let's use k=0.7 as a balance
    k = 0.7
    cost_gate = np.exp(-k * cost)
    
    # Score is EI modulated by exponential cost gate
    scores = ei * cost_gate
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
