import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Driven Duffing oscillator with complex-exponential state representation.
    
    Uses z = x + i*v to naturally couple position and velocity.
    The acceleration is modeled as:
      a = Re[ p0 * z^3 + p1 * z^2 + p2 * z + p3 * conj(z) 
            + (p4 + i*p5) * exp(i*t) 
            + (p6 + i*p7) * z * exp(i*t) 
            + p8 * |v| * v + p9 ]
    
    This structure:
    - Uses complex polynomial in z = x + iv to capture x^3, x^2v, xv^2, v^3, x^2, xv, v^2
    - Uses complex exponential for driving (captures sin(t) and cos(t) with phase)
    - Uses parametric drive z*exp(i*t) for x*sin(t), x*cos(t), v*sin(t), v*cos(t)
    - Includes nonlinear damping |v|*v
    - 10 parameters total
    """
    p = params
    
    # Complex state
    z = x + 1j * v
    z_conj = x - 1j * v
    
    # Powers of z
    z2 = z * z
    z3 = z2 * z
    
    # Complex exponential (driving)
    exp_it = np.exp(1j * t)
    
    # Parametric drive
    z_exp = z * exp_it
    
    # Nonlinear damping (real-valued)
    v_abs = np.abs(v)
    v_quad = v * v_abs
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p
    
    # Complex-valued drive terms
    drive1 = (p4 + 1j * p5) * exp_it
    drive2 = (p6 + 1j * p7) * z_exp
    
    # Polynomial in z (complex)
    poly = p0 * z3 + p1 * z2 + p2 * z + p3 * z_conj
    
    # Combine: real part of complex terms + real nonlinear terms
    result = np.real(poly + drive1 + drive2) + p8 * v_quad + p9
    
    return result
