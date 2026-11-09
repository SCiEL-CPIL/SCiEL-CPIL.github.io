import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using Chebyshev polynomial basis for state variables
    and parametric sinusoidal drive.
    
    Model:
    a = p0 * sin(p1 * t + p2) + p3 * T1(xn) + p4 * T2(xn) + p5 * T3(xn) 
      + p6 * T1(vn) + p7 * T3(vn) + p8 * xn * vn + p9
    
    Where xn, vn are normalized x, v to [-1, 1].
    T1(z) = z
    T2(z) = 2z^2 - 1
    T3(z) = 4z^3 - 3z
    
    Parameters:
    p0: Drive amplitude
    p1: Drive frequency
    p2: Drive phase
    p3: Linear stiffness (Chebyshev T1)
    p4: Quadratic stiffness (Chebyshev T2)
    p5: Cubic stiffness (Chebyshev T3)
    p6: Linear damping (Chebyshev T1)
    p7: Cubic damping (Chebyshev T3)
    p8: Position-velocity coupling
    p9: Constant bias
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p
    
    # Normalize x and v to [-1, 1] for Chebyshev evaluation
    # x range: [-0.2056, 0.1749] -> center = -0.01535, half_range = 0.19025
    # v range: [-0.3391, 0.3624] -> center = 0.01165, half_range = 0.35075
    x_center = -0.01535
    x_half = 0.19025
    v_center = 0.01165
    v_half = 0.35075
    
    xn = (x - x_center) / x_half
    vn = (v - v_center) / v_half
    
    # Clip to [-1, 1] to ensure stability
    xn = np.clip(xn, -1.0, 1.0)
    vn = np.clip(vn, -1.0, 1.0)
    
    # Chebyshev polynomials
    T1_x = xn
    T2_x = 2.0 * xn**2 - 1.0
    T3_x = 4.0 * xn**3 - 3.0 * xn
    
    T1_v = vn
    T3_v = 4.0 * vn**3 - 3.0 * vn
    
    # Drive term
    drive = p0 * np.sin(p1 * t + p2)
    
    # State terms
    state = (p3 * T1_x + p4 * T2_x + p5 * T3_x + 
             p6 * T1_v + p7 * T3_v + 
             p8 * xn * vn)
    
    result = drive + state + p9
    
    # Safety: replace non-finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
