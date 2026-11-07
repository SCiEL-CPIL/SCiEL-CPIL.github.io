"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for acceleration of a nonlinear oscillator.
    Strategy: Physically constrained autonomous model (Duffing + Van der Pol).
    
    The model assumes the dynamics are governed by:
    1. Stiffness: Linear (p0*x) and Cubic (p1*x^3) terms.
    2. Damping: Linear (p2*v) and Cubic saturation (p3*v^3) terms.
    3. State-dependent damping: (p4*x^2)*v term (Van der Pol style).
    4. Higher-order stiffness: p5*x^5 for better limit cycle shape.
    5. Nuisance/Residual terms: p6, p7, p8, p9 for flexibility.
    
    We use 10 parameters.
    p0: Linear stiffness (k)
    p1: Cubic stiffness (alpha)
    p2: Linear damping (c)
    p3: Cubic damping (beta) - usually negative for limit cycle onset
    p4: Quadratic position * velocity (gamma*x^2*v) - Van der Pol term
    p5: Quintic stiffness (x^5)
    p6: x*v term
    p7: x^2*v term
    p8: v^2 term (asymmetric damping?)
    p9: Constant offset (bias)
    """
    p = params
    
    # Core Physics Terms
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    
    v2 = v * v
    v3 = v2 * v
    
    # Duffing stiffness
    term_stiff_lin = p[0] * x
    term_stiff_cub = p[1] * x3
    term_stiff_quin = p[5] * x5
    
    # Damping terms
    term_damp_lin = p[2] * v
    term_damp_cub = p[3] * v3
    # Van der Pol style: damping depends on x^2
    term_damp_vdp = p[4] * x2 * v
    
    # Additional flexibility terms
    term_xv = p[6] * x * v
    term_x2v = p[7] * x2 * v
    term_v2 = p[8] * v2
    term_bias = p[9]
    
    # Sum
    y = (term_stiff_lin + term_stiff_cub + term_stiff_quin + 
         term_damp_lin + term_damp_cub + term_damp_vdp +
         term_xv + term_x2v + term_v2 + term_bias)
         
    return y

# EVOLVE-BLOCK-END
