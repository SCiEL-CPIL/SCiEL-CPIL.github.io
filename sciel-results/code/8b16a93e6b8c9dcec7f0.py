"""Symbolic regression with polynomial + Gaussian bump basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure we have exactly 10 parameters; pad with zeros if needed
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Polynomial terms up to degree 5 (6 coefficients)
    poly = (p[0] + p[1] * x + p[2] * x**2 + p[3] * x**3 
            + p[4] * x**4 + p[5] * x**5)
    
    # Gaussian bumps at fixed centers with learnable amplitudes
    # Centers: -1.5, -0.5, 0.5, 1.5; fixed sigma = 0.8
    sigma = 0.8
    centers = np.array([-1.5, -0.5, 0.5, 1.5])
    
    gauss_sum = np.zeros_like(x)
    for i, c in enumerate(centers):
        gauss_sum += p[6 + i] * np.exp(-0.5 * ((x - c) / sigma) ** 2)
    
    return poly + gauss_sum
# EVOLVE-BLOCK-END
