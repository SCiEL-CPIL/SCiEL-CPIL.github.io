"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator using a physics-informed
    Duffing-van der Pol structure with scaled inputs.
    
    The model uses a basis motivated by nonlinear oscillator physics:
    - Duffing potential: x + x^3 (restoring force)
    - Van der Pol damping: v, v^3 (nonlinear damping)
    - Coupling terms: x*v^2, v*x^2, x^2*v, x*v
    - Higher order: x^4, v^4, x^2*v^2, constant
    
    Inputs are scaled to order 1 to improve conditioning for BFGS.
    """
    # Fixed scaling constants derived from the training data range
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    # Scale inputs to approximately [-1, 1]
    xs = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    vs = 2.0 * (v - v_min) / (v_max - v_min) - 1.0
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Physics-informed Duffing-van der Pol structure
    # Restoring force terms (Duffing)
    restoring = p0 * xs + p1 * xs**3 + p2 * xs**5
    
    # Damping terms (van der Pol style)
    damping = p3 * vs + p4 * vs**3
    
    # Coupling terms
    coupling = p5 * xs * vs**2 + p6 * vs * xs**2 + p7 * xs**2 * vs**2
    
    # Additional higher-order terms
    higher = p8 * xs**4 + p9 * vs**4
    
    return restoring + damping + coupling + higher
# EVOLVE-BLOCK-END
