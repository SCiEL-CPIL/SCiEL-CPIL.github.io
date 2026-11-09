"""Symbolic regression with rich basis and robust BFGS fitting."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Build a well-conditioned basis:
    # p[0]: constant
    # p[1]: x
    # p[2]: x^2
    # p[3]: x^3
    # p[4]: sin(x)
    # p[5]: cos(x)
    # p[6]: sin(2x)
    # p[7]: cos(2x)
    # p[8]: exp(-x^2)  (Gaussian centered at 0)
    # p[9]: exp(-((x-1)^2)) (Gaussian centered at 1)
    
    term0 = np.ones_like(x)
    term1 = x
    term2 = x ** 2
    term3 = x ** 3
    term4 = np.sin(x)
    term5 = np.cos(x)
    term6 = np.sin(2.0 * x)
    term7 = np.cos(2.0 * x)
    term8 = np.exp(-x ** 2)
    term9 = np.exp(-((x - 1.0) ** 2))
    
    return (p[0] * term0 + p[1] * term1 + p[2] * term2 + p[3] * term3 +
            p[4] * term4 + p[5] * term5 + p[6] * term6 + p[7] * term7 +
            p[8] * term8 + p[9] * term9)
# EVOLVE-BLOCK-END
