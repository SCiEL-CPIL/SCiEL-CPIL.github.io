import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Parametric limit-cycle oscillator model.
    
    a = -k(x)*x - c(v)*v + F(t)
    
    where:
    k(x) = p0 + p1*tanh(x) + p2*x^2      (position-dependent stiffness)
    c(v) = p3 + p4*tanh(v) + p5*v*|v|    (velocity-dependent damping)
    F(t) = p6*cos(t) + p7*sin(t)          (periodic driving)
    plus: p8*x*v (parametric coupling) + p9 (offset)
    
    All 10 parameters are fitted.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Position-dependent stiffness: k(x) = p0 + p1*tanh(x) + p2*x^2
    k_x = p0 + p1 * np.tanh(x) + p2 * x * x
    
    # Velocity-dependent damping: c(v) = p3 + p4*tanh(v) + p5*v*|v|
    c_v = p3 + p4 * np.tanh(v) + p5 * v * np.abs(v)
    
    # Periodic driving force
    F_t = p6 * np.cos(t) + p7 * np.sin(t)
    
    # Parametric coupling term
    coupling = p8 * x * v
    
    # Offset
    offset = p9
    
    # Acceleration: a = -k(x)*x - c(v)*v + F(t) + coupling + offset
    return -k_x * x - c_v * v + F_t + coupling + offset
