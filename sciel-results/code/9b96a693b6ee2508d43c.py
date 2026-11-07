import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Duffing oscillator with parametric forcing.
    
    Model: a = -k(t)*x - c*x^3 - d*v + F0*sin(w*t + phi)
    where k(t) = k0 + k1*cos(w2*t) captures parametric excitation.
    
    Parameters (10):
    p0: k0 (linear stiffness DC)
    p1: k1 (parametric stiffness modulation amplitude)
    p2: c (cubic stiffness, Duffing term)
    p3: d (damping coefficient)
    p4: F0 (forcing amplitude)
    p5: w (forcing frequency)
    p6: phi (forcing phase)
    p7: w2 (parametric modulation frequency)
    p8: x_scale (normalization for x)
    p9: v_scale (normalization for v)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Safe normalizations with fallbacks
    x_s = np.where(np.abs(p8) > 1e-6, p8, 1.0)
    v_s = np.where(np.abs(p9) > 1e-6, p9, 1.0)
    
    xn = x / x_s
    vn = v / v_s
    
    # Parametric stiffness modulation
    k_t = p0 + p1 * np.cos(p7 * t)
    
    # Duffing oscillator terms
    accel = -k_t * xn - p2 * (xn ** 3) - p3 * vn
    
    # External periodic forcing
    accel += p4 * np.sin(p5 * t + p6)
    
    return accel
