"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model.
    
    Strategy: Physics-informed nonlinear basis.
    Instead of a linear combination of monomials (which is ill-conditioned),
    we use a structure that mirrors the Van der Pol and Duffing equations.
    
    The acceleration a = dv/dt is modeled as:
    a = -k_eff(x) * x - c_eff(x) * v + bias
    
    Where:
    - k_eff(x) = p0 + p1*x^2  (Duffing-like nonlinear stiffness)
    - c_eff(x) = p2 + p3*x^2  (Van der Pol-like nonlinear damping)
    
    To utilize all 10 parameters and allow for more complex dynamics if needed,
    we expand the effective coefficients with higher-order terms but keep them
    structurally grouped to maintain conditioning:
    
    a = -(p0 + p1*x^2 + p2*x^4) * x 
        - (p3 + p4*x^2 + p5*x^4) * v 
        + p6*x^3 + p7*v^2 + p8*x*v + p9
    
    This form is:
    1. Structurally aligned with known oscillator physics.
    2. Less collinear than a pure polynomial basis because the parameters 
       control the *shape* of the force-velocity/displacement curves.
    3. BFGS-friendly: The gradients with respect to p0..p9 are well-behaved 
       and distinct features.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    v2 = v * v
    xv = x * v
    
    # Nonlinear stiffness term: F_restoring = -(p0 + p1*x^2 + p2*x^4) * x
    # This covers x, x^3, x^5 terms but parameterized by stiffness coefficients
    restoring_force = -(p0 * x + p1 * x3 + p2 * x * x4)
    
    # Nonlinear damping term: F_damping = -(p3 + p4*x^2 + p5*x^4) * v
    # This covers v, x^2*v, x^4*v terms
    damping_force = -(p3 * v + p4 * x2 * v + p5 * x4 * v)
    
    # Additional higher-order corrections to allow for asymmetry or other effects
    # These are kept small in effect if the main physics terms fit well
    correction = p6 * x3 + p7 * v2 + p8 * xv + p9
    
    return restoring_force + damping_force + correction

# EVOLVE-BLOCK-END
