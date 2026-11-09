"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s limit


def _elapsed_ok():
    return (time.time() - _START_TIME) < TIME_LIMIT


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Non-linear parametric model: sum of 2 Gaussian bumps + interaction term.
    
    Structure:
      pred = p0 + p1*exp(-p2*((x-p3)^2 + (y-p4)^2)) 
                   + p5*exp(-p6*((x-p7)^2 + (y-p8)^2)) 
                   + p9*x*y
    
    This is a different modeling family from linear feature bases:
    - Parameters define centers, widths, and amplitudes of basis functions
    - Non-linear in parameters (widths/centers inside exp)
    - Can capture localized structures and interactions
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Clamp widths to avoid numerical issues (widths should be positive)
    w1 = np.clip(p2, 0.01, 10.0)
    w2 = np.clip(p6, 0.01, 10.0)
    
    # Gaussian 1: centered at (p3, p4), width w1
    dx1 = x - p3
    dy1 = y - p4
    g1 = np.exp(-w1 * (dx1**2 + dy1**2))
    
    # Gaussian 2: centered at (p7, p8), width w2
    dx2 = x - p7
    dy2 = y - p8
    g2 = np.exp(-w2 * (dx2**2 + dy2**2))
    
    # Interaction term
    xy = x * y
    
    pred = p0 + p1 * g1 + p5 * g2 + p9 * xy
    
    # Ensure finite values
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
# EVOLVE-BLOCK-END
