"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a physics-informed basis.
    
    The model assumes the form of a driven Duffing/Van der Pol oscillator:
    a = p0 + p1*x + p2*x^3 + p3*v + p4*v^3 + p5*x*v + p6*sin(t) + p7*cos(t) + p8*t + p9
    
    This covers:
    - p1*x: Linear spring (harmonic oscillator)
    - p2*x^3: Duffing nonlinearity (hard/soft spring)
    - p3*v: Linear damping
    - p4*v^3: Van der Pol nonlinearity (self-excitation)
    - p5*x*v: Parametric or cross-coupling term
    - p6*sin(t), p7*cos(t): Periodic forcing
    - p8*t: Secular drift (often 0 in steady state, but allows transient)
    - p9: Bias/offset
    
    This structure is much closer to the underlying physics than a generic NN,
    which should help BFGS converge to a meaningful solution even with a simple
    initial guess of all ones.
    """
    start_time = time.time()
    
    # Unpack parameters
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Compute basis terms
    # Note: We use np.sin and np.cos for periodic forcing
    # x^3 and v^3 can be large, but inputs are bounded [-0.33, 0.36], so x^3 ~ 0.04
    # v^3 ~ 0.04. No overflow risk.
    
    term_x = x
    term_x3 = x * x * x
    term_v = v
    term_v3 = v * v * v
    term_xv = x * v
    term_sin_t = np.sin(t)
    term_cos_t = np.cos(t)
    term_t = t
    
    # Combine
    result = (p0 + 
              p1 * term_x + 
              p2 * term_x3 + 
              p3 * term_v + 
              p4 * term_v3 + 
              p5 * term_xv + 
              p6 * term_sin_t + 
              p7 * term_cos_t + 
              p8 * term_t + 
              p9)
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(t)
        
    return result
# EVOLVE-BLOCK-END
