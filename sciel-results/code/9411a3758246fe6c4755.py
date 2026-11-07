"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate IQR of the predicted means to estimate uncertainty in the landscape
    # This serves as a proxy for the spread of the function values
    q1, q3 = np.percentile(mu, [25, 75])
    iqr_mu = q3 - q1
    
    # Adaptive xi: negative value encourages exploration by lowering the 
    # improvement threshold relative to the spread of predictions.
    # Points with high variance (sigma) become more attractive if the 
    # threshold is lowered, as their tails extend further into the 
    # "improvement" region.
    k = 0.1
    xi = -k * iqr_mu
    
    # Clip xi to prevent it from becoming too negative, which could 
    # lead to excessive random exploration if the model is poor.
    xi = np.clip(xi, -5.0, 0.0)
    
    # Expected Improvement with adaptive xi
    # For minimization, EI is based on f_best + xi
    target = f_best + xi
    z = (target - mu) / sigma
    
    # Compute EI
    ei = (target - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Handle numerical issues
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return ei
# EVOLVE-BLOCK-END
