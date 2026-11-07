# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Parametric oscillator model with time-varying coefficients via low-order Fourier series.
    
    Design: State-space representation where stiffness and damping are modulated
    by periodic functions of t, capturing parametric excitation (Mathieu-type dynamics).
    
    Model:
      a = (k0 + k1*cos(t) + k2*sin(t)) * x          [time-varying stiffness]
        + (c0 + c1*cos(t)) * v                       [time-varying damping]
        + alpha * x^3                                 [Duffing nonlinearity]
        + F0*cos(t) + F1*sin(t)                      [external forcing]
        + beta * x^2                                  [quadratic asymmetry]
        + gamma * v^2                                 [quadratic velocity effect]
        + bias
    
    This uses exactly 10 parameters:
      p0 = k0   (mean stiffness)
      p1 = k1   (stiffness modulation, cos)
      p2 = k2   (stiffness modulation, sin)
      p3 = c0   (mean damping)
      p4 = c1   (damping modulation, cos)
      p5 = alpha (Duffing cubic stiffness)
      p6 = F0   (forcing, cos)
      p7 = F1   (forcing, sin)
      p8 = beta  (x^2 coefficient)
      p9 = gamma (v^2 coefficient)
    
    All terms are O(1) given input ranges, providing good conditioning for BFGS.
    """
    deadline = time.time() + 145.0  # 5s safety margin before 150s kill
    
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
    
    # Precompute trigonometric functions
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    
    # Time-varying stiffness: (p0 + p1*cos(t) + p2*sin(t)) * x
    k_t = p0 + p1 * cos_t + p2 * sin_t
    term_stiffness = k_t * x
    
    # Time-varying damping: (p3 + p4*cos(t)) * v
    c_t = p3 + p4 * cos_t
    term_damping = c_t * v
    
    # Duffing nonlinearity
    term_duffing = p5 * x * x * x
    
    # External forcing
    term_forcing = p6 * cos_t + p7 * sin_t
    
    # Quadratic terms
    term_x2 = p8 * x * x
    term_v2 = p9 * v * v
    
    # Combine all terms
    result = term_stiffness + term_damping + term_duffing + term_forcing + term_x2 + term_v2
    
    # Safety check with deadline guard
    if not np.all(np.isfinite(result)):
        if time.time() > deadline:
            return np.zeros_like(t)
        return np.zeros_like(t)
    
    return result
# EVOLVE-BLOCK-END
