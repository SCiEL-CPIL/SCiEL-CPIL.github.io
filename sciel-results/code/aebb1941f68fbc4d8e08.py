import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a physically-motivated basis
    with normalized inputs for better BFGS conditioning.
    
    The basis combines:
    - Position terms: x, x^2, x^3 (restoring force, Duffing-type)
    - Velocity terms: v, v*|v| (linear and quadratic damping)
    - Cross term: x*v (parametric coupling)
    - Time terms: sin(t), cos(t) (periodic driving)
    - Constant offset
    
    Inputs are normalized to improve numerical conditioning for BFGS.
    """
    # Normalize inputs to approximately [-1, 1]
    # t: [30, 50] -> center 40, scale 10
    # x: [-0.2056, 0.1749] -> center -0.015, scale 0.2
    # v: [-0.3391, 0.3624] -> center 0.01, scale 0.35
    t_n = (t - 40.0) / 10.0
    x_n = (x + 0.015) / 0.2
    v_n = (v - 0.01) / 0.35
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis functions
    x2 = x_n * x_n
    x3 = x2 * x_n
    v_abs = np.abs(v_n)
    v_quad = v_n * v_abs  # quadratic damping (odd power)
    xv = x_n * v_n
    St = np.sin(t)
    Ct = np.cos(t)
    
    # Linear combination of basis functions
    # p0: constant
    # p1: linear restoring (x)
    # p2: quadratic restoring (x^2)
    # p3: cubic restoring (x^3) - Duffing
    # p4: linear damping (v)
    # p5: quadratic damping (v*|v|)
    # p6: cross coupling (x*v)
    # p7: sinusoidal drive
    # p8: cosinusoidal drive
    # p9: linear time drift (t_n)
    res = (p0 + 
           p1 * x_n + 
           p2 * x2 + 
           p3 * x3 + 
           p4 * v_n + 
           p5 * v_quad + 
           p6 * xv + 
           p7 * St + 
           p8 * Ct + 
           p9 * t_n)
    
    return res
