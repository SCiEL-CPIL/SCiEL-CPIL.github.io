import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Parametric limit-cycle oscillator with bounded nonlinearities.
    
    a = -k(x)*x - c(v)*v + F(t) + coupling + offset
    
    where:
    k(x) = p0 + p1*tanh(x)          (bounded, smooth stiffness)
    c(v) = p2 + p3*tanh(v)          (bounded, smooth damping)
    F(t) = p4*cos(t) + p5*sin(t)    (periodic driving)
    coupling = p6*x*v + p7*x^2 + p8*v^2  (energy exchange + quadratic terms)
    offset = p9
    
    The tanh functions bound the stiffness and damping, preventing BFGS from
    running into unbounded polynomial growth and keeping the optimization
    landscape well-conditioned. The quadratic terms x^2 and v^2 add
    necessary nonlinearity for the limit-cycle dynamics.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Bounded, smooth position-dependent stiffness
    k_x = p0 + p1 * np.tanh(x)
    
    # Bounded, smooth velocity-dependent damping
    c_v = p2 + p3 * np.tanh(v)
    
    # Periodic driving force
    F_t = p4 * np.cos(t) + p5 * np.sin(t)
    
    # Coupling and quadratic terms
    coupling = p6 * x * v + p7 * x * x + p8 * v * v
    
    # Offset
    offset = p9
    
    # Acceleration
    return -k_x * x - c_v * v + F_t + coupling + offset
