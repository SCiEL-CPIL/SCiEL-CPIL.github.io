"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Basis functions
    # 0: 1
    # 1: x
    # 2: y
    # 3: x^2
    # 4: y^2
    # 5: x*y
    # 6: x^3
    # 7: y^3
    # 8: x^2*y
    # 9: x*y^2
    
    term0 = np.ones_like(x)
    term1 = x
    term2 = y
    term3 = x * x
    term4 = y * y
    term5 = x * y
    term6 = x * x * x
    term7 = y * y * y
    term8 = x * x * y
    term9 = x * y * y
    
    return (params[0] * term0 +
            params[1] * term1 +
            params[2] * term2 +
            params[3] * term3 +
            params[4] * term4 +
            params[5] * term5 +
            params[6] * term6 +
            params[7] * term7 +
            params[8] * term8 +
            params[9] * term9)
# EVOLVE-BLOCK-END
