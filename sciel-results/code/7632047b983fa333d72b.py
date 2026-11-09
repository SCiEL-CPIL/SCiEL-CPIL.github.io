"""Rewritten symbolic regression program with richer basis and robust fitting."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has at least 10 elements; pad with zeros if needed
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Build basis functions
    # Polynomial terms up to degree 4
    term0 = np.ones_like(x)
    term1 = x
    term2 = x**2
    term3 = x**3
    term4 = x**4
    
    # Gaussian terms at fixed centers
    c0 = -2.0
    c1 = -1.0
    c2 = 0.0
    c3 = 1.0
    c4 = 2.0
    
    # Use moderate width to capture local features
    sigma = 0.8
    term5 = np.exp(-((x - c0)**2) / (2 * sigma**2))
    term6 = np.exp(-((x - c1)**2) / (2 * sigma**2))
    term7 = np.exp(-((x - c2)**2) / (2 * sigma**2))
    term8 = np.exp(-((x - c3)**2) / (2 * sigma**2))
    term9 = np.exp(-((x - c4)**2) / (2 * sigma**2))
    
    # Combine
    y = (p[0] * term0 + p[1] * term1 + p[2] * term2 + p[3] * term3 + p[4] * term4 +
         p[5] * term5 + p[6] * term6 + p[7] * term7 + p[8] * term8 + p[9] * term9)
    
    return y
# EVOLVE-BLOCK-END
