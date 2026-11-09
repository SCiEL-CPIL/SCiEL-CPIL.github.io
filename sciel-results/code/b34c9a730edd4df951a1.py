import numpy as np
from scipy.special import ndtr

def acquisition(mu, sigma, f_best):
    sigma = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma
    Phi_z = ndtr(z)
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
    phi_z = np.exp(-0.5 * z * z) * inv_sqrt_2pi
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Add exploration term
    exploration_weight = 0.5
    scores = ei + exploration_weight * sigma
    
    return scores
