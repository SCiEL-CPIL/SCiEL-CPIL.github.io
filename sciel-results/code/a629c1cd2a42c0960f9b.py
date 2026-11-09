import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Structural Change: Radial Basis Functions + Polynomial Background
    # This allows capturing local non-linear features (RBFs) and global trends (Poly).
    
    # RBF Centers (fixed at corners of the domain)
    c1x, c1y = -0.5, -0.5
    c2x, c2y =  0.5, -0.5
    c3x, c3y = -0.5,  0.5
    c4x, c4y =  0.5,  0.5
    
    # RBF Width (fixed)
    # Using a width that allows significant overlap between centers
    sigma_sq = 0.5
    
    # Compute RBF terms
    d1_sq = (x - c1x)**2 + (y - c1y)**2
    d2_sq = (x - c2x)**2 + (y - c2y)**2
    d3_sq = (x - c3x)**2 + (y - c3y)**2
    d4_sq = (x - c4x)**2 + (y - c4y)**2
    
    r1 = np.exp(-d1_sq / sigma_sq)
    r2 = np.exp(-d2_sq / sigma_sq)
    r3 = np.exp(-d3_sq / sigma_sq)
    r4 = np.exp(-d4_sq / sigma_sq)
    
    # Polynomial Background Terms
    p0 = np.ones_like(x)
    p1 = x
    p2 = y
    p3 = x**2
    p4 = y**2
    p5 = x * y
    
    return (params[0] * r1 +
            params[1] * r2 +
            params[2] * r3 +
            params[3] * r4 +
            params[4] * p0 +
            params[5] * p1 +
            params[6] * p2 +
            params[7] * p3 +
            params[8] * p4 +
            params[9] * p5)
