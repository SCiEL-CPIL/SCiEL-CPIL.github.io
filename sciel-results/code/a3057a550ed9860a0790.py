"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
START_TIME = time.time()
DEADLINE = 120.0  # 120s limit, keep margin

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a physics-informed Duffing-van der Pol basis.
    
    Basis terms:
    1. Constant (p0)
    2. Linear restoring (p1 * x)
    3. Nonlinear restoring cubic (p2 * x^3)
    4. Linear damping (p3 * v)
    5. Nonlinear damping cubic (p4 * v^3)
    6. Velocity-squared stiffness coupling (p5 * x * v^2)
    7. Position-squared velocity coupling (p6 * x^2 * v)  <-- Van der Pol term
    8. Quartic restoring (p7 * x^4)
    9. Quartic damping (p8 * v^4)
    10. Higher order coupling (p9 * x^2 * v^2)
    
    Note: The prompt suggested a specific order. I will map params to these terms.
    The prompt's suggested structure:
    a = p_1 x + p_2 x^3 - p_3 v - p_4 v^3 + p_5 x v^2 + p_6 v^2 x^2 + p_7 x^2 v + p_8 x^4 + p_9 v^4 + p_{10}
    
    I will use this exact structure but ensure all parameters are free (no fixed signs) to maximize flexibility for BFGS, 
    although the structure itself guides the basis selection.
    """
    # Check deadline
    if time.time() > DEADLINE:
        # Return a neutral prediction if we are running out of time
        # This shouldn't happen during fitting, but good for safety
        return np.zeros_like(x)

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Compute powers
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2

    # Terms based on Duffing-van der Pol physics
    # p1: Linear restoring (x)
    # p2: Nonlinear restoring (x^3)
    # p3: Linear damping (v)
    # p4: Nonlinear damping (v^3)
    # p5: Coupling x * v^2
    # p6: Coupling x^2 * v (Van der Pol style)
    # p7: Quartic restoring (x^4)
    # p8: Quartic damping (v^4)
    # p9: Coupling x^2 * v^2
    # p0: Bias
    
    term1 = p1 * x
    term2 = p2 * x3
    term3 = p3 * v
    term4 = p4 * v3
    term5 = p5 * x * v2
    term6 = p6 * x2 * v
    term7 = p7 * x4
    term8 = p8 * v4
    term9 = p9 * x2 * v2
    
    return p0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
# EVOLVE-BLOCK-END
