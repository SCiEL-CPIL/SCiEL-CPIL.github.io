import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    State-space oscillator with time-varying coefficients via low-order Fourier series.
    
    The model uses a Duffing-type nonlinear oscillator where the effective stiffness
    and damping coefficients vary periodically with time via Fourier series.
    
    Structure:
    a(t, x, v) = [k0 + k1*sin(t) + k2*cos(t)] * x
              + [k3 + k4*sin(t) + k5*cos(t)] * x^3
              + [c0 + c1*sin(t) + c2*cos(t)] * v
              + [f0 + f1*sin(t) + f2*cos(t)]
    
    This gives 10 parameters total:
    p0-p2: constant, sin(t), cos(t) coefficients for linear stiffness
    p3-p5: constant, sin(t), cos(t) coefficients for cubic stiffness
    p6-p8: constant, sin(t), cos(t) coefficients for linear damping
    p9:    constant forcing term (absorbed into bias)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute time-dependent coefficients
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Time-varying linear stiffness: k(t) = p0 + p1*sin(t) + p2*cos(t)
    k_t = p0 + p1 * sin_t + p2 * cos_t
    
    # Time-varying cubic stiffness: k3(t) = p3 + p4*sin(t) + p5*cos(t)
    k3_t = p3 + p4 * sin_t + p5 * cos_t
    
    # Time-varying linear damping: c(t) = p6 + p7*sin(t) + p8*cos(t)
    c_t = p6 + p7 * sin_t + p8 * cos_t
    
    # Constant forcing term
    f0 = p9
    
    # State variables
    x2 = x * x
    x3 = x2 * x
    
    # Combine: a = k(t)*x + k3(t)*x^3 + c(t)*v + f0
    return k_t * x + k3_t * x3 + c_t * v + f0
