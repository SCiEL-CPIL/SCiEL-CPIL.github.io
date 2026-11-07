"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    
    # Hybrid basis: Polynomials for trends, Trig for periodicity, Gaussian for localization
    # 1. Constant (DC)
    # 2-3. Low order polynomials (normalized to avoid ill-conditioning)
    # 4-9. Sines and Cosines for freq 1 and 2
    # 10. Gaussian bump
    
    t = x / 3.0  # Normalize to [-1, 1]
    
    basis = np.column_stack([
        np.ones_like(t),
        t,
        t * t,
        np.sin(x),
        np.cos(x),
        np.sin(2.0 * x),
        np.cos(2.0 * x),
        np.sin(3.0 * x),
        np.cos(3.0 * x),
        np.exp(-t * t)
    ])
    
    return basis @ params
# EVOLVE-BLOCK-END
