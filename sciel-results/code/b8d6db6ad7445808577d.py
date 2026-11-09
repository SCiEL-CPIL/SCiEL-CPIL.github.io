import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Polynomial-based nonlinear oscillator model with higher-order terms.
    
    This structural change moves from tanh-based bounded nonlinearities to a
    pure polynomial representation, which may better capture the underlying
    dynamics if the system is well-approximated by polynomials.
    
    Model:
    a = p0 + p1*x + p2*v + p3*x^2 + p4*v^2 + p5*x*v + p6*x^3 + p7*v^3 + 
        p8*sin(p9*t) * x + p10*cos(p9*t) * v
        
    Note: We use 10 parameters total. The drive frequency is shared between 
    sine and cosine terms for efficiency.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial terms in x and v
    term0 = p0
    term1 = p1 * x
    term2 = p2 * v
    term3 = p3 * x**2
    term4 = p4 * v**2
    term5 = p5 * x * v
    term6 = p6 * x**3
    term7 = p7 * v**3
    
    # Time-dependent drive terms coupled to state variables
    drive_freq = p9
    drive_sin = np.sin(drive_freq * t)
    drive_cos = np.cos(drive_freq * t)
    
    term8 = p8 * drive_sin * x
    term9 = p8 * drive_cos * v  # Reuse p8 for both drive amplitudes with different phases
    
    res = term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
    
    return res
