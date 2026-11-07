"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Piecewise-linear (PL) basis model for stress-strain of an aluminium rod.
    
    Structural change from global polynomial to local piecewise-linear basis.
    This captures the elastic-plastic transition (yield point) and strain
    hardening more effectively than a smooth polynomial, which may oscillate
    or underfit sharp transitions.
    
    The model is linear in the 10 parameters, allowing BFGS to converge
    reliably to the global minimum for this basis.
    
    Basis functions:
    - Strain part: 1, s, ReLU(s-k1), ReLU(s-k2), ReLU(s-k3)
    - Temp part: t, s*t, ReLU(s-k1)*t, ReLU(s-k2)*t, ReLU(s-k3)*t
    
    Fixed knots k1=0.01 (yield), k2=0.1 (hardening start), k3=0.5 (hardening plateau)
    """
    s = strain
    t = temp
    
    # Fixed knots based on typical aluminium stress-strain behavior
    k1 = 0.01
    k2 = 0.1
    k3 = 0.5
    
    # Strain basis functions
    b0 = np.ones_like(s)
    b1 = s
    b2 = np.maximum(0.0, s - k1)
    b3 = np.maximum(0.0, s - k2)
    b4 = np.maximum(0.0, s - k3)
    
    # Temp interaction basis functions
    b5 = t
    b6 = s * t
    b7 = b2 * t
    b8 = b3 * t
    b9 = b4 * t
    
    return (params[0] * b0 + params[1] * b1 + params[2] * b2 + params[3] * b3 + params[4] * b4 +
            params[5] * b5 + params[6] * b6 + params[7] * b7 + params[8] * b8 + params[9] * b9)
# EVOLVE-BLOCK-END
