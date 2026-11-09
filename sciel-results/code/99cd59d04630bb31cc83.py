import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Parametric limit-cycle oscillator structure:
    a = -(1 + p0*tanh(x)) * x - (1 + p1*tanh(v)) * v + p2*sin(t) + p3*cos(t)
      + p4*x^2 + p5*v^2 + p6*x*v + p7*tanh(x)*tanh(v) + p8*sin(2*t) + p9*cos(2*t)
    
    Uses bounded nonlinearities (tanh) for stable dynamics.
    """
    # Extract params
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Bounded nonlinearities
    tanh_x = np.tanh(x)
    tanh_v = np.tanh(v)
    
    # Powers
    x2 = x * x
    v2 = v * v
    
    # Trig
    St = np.sin(t)
    Ct = np.cos(t)
    S2t = np.sin(2.0 * t)
    C2t = np.cos(2.0 * t)
    
    # Stiffness and damping modulations
    k_eff = 1.0 + p0 * tanh_x
    c_eff = 1.0 + p1 * tanh_v
    
    # Compute acceleration
    res = (-k_eff * x - 
           c_eff * v + 
           p2 * St + 
           p3 * Ct + 
           p4 * x2 + 
           p5 * v2 + 
           p6 * x * v + 
           p7 * tanh_x * tanh_v + 
           p8 * S2t + 
           p9 * C2t)
           
    return res
