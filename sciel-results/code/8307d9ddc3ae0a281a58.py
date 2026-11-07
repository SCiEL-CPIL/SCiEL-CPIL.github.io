import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration modeled with a physics-informed basis
    including state-dependent terms and parametric time modulation.
    
    Inputs:
    t: time
    x: position
    v: velocity
    params: array of 10 fitted constants
    
    Returns:
    y: acceleration
    """
    # Scale inputs to O(1) to improve conditioning for BFGS
    # Based on training ranges: x ~ [-0.21, 0.17], v ~ [-0.34, 0.36]
    x_scale = 0.2
    v_scale = 0.3
    
    xs = x / x_scale
    vs = v / v_scale
    
    # Trigonometric functions of time
    st = np.sin(t)
    ct = np.cos(t)
    
    # Basis functions
    # State terms
    b0 = xs          # Linear position
    b1 = xs * xs     # Quadratic position
    b2 = xs * xs * xs # Cubic position (Duffing)
    b3 = vs          # Linear velocity (Damping)
    b4 = vs * vs     # Quadratic velocity
    b5 = xs * vs     # Cross term
    
    # Time forcing terms
    b6 = st          # Sinusoidal forcing
    b7 = ct          # Cosinusoidal forcing
    
    # Parametric terms (Time-varying coefficients for x)
    b8 = xs * st     # x modulated by sin(t)
    b9 = xs * ct     # x modulated by cos(t)
    
    # Combine with parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    result = (p0 * b0 + 
              p1 * b1 + 
              p2 * b2 + 
              p3 * b3 + 
              p4 * b4 + 
              p5 * b5 + 
              p6 * b6 + 
              p7 * b7 + 
              p8 * b8 + 
              p9 * b9)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(t, dtype=np.float64)
        
    return result
