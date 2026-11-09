import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression for a nonlinear oscillator: a = f(x, v, t)
    
    Uses Fourier series in time for periodic driving and Chebyshev polynomials
    in position for the restoring force, with tanh-bounded nonlinearities to
    provide a well-conditioned, near-convex landscape for BFGS.
    
    Parameters (10 total):
    p0: bias
    p1: Chebyshev T1(x) coefficient (linear restoring)
    p2: Chebyshev T2(x) coefficient (quadratic restoring)
    p3: Chebyshev T3(x) coefficient (cubic restoring)
    p4: linear damping coefficient
    p5: nonlinear damping coefficient (tanh-bounded)
    p6: sin(t) forcing amplitude
    p7: cos(t) forcing amplitude
    p8: sin(2t) forcing amplitude (2nd harmonic)
    p9: cos(2t) forcing amplitude (2nd harmonic)
    """
    # Normalize inputs to [-1, 1] range for Chebyshev stability
    # x: [-0.2056, 0.1749] -> center -0.01535, half-range 0.19025
    x_center = -0.01535
    x_half = 0.19025
    x_n = (x - x_center) / x_half
    
    # v: [-0.3391, 0.3624] -> center 0.01165, half-range 0.35075
    v_center = 0.01165
    v_half = 0.35075
    v_n = (v - v_center) / v_half
    
    # Clamp to [-1, 1] for Chebyshev stability
    x_c = np.clip(x_n, -1.0, 1.0)
    v_c = np.clip(v_n, -1.0, 1.0)
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Chebyshev polynomials in x (position-dependent restoring force)
    T0 = np.ones_like(x_c)
    T1 = x_c
    T2 = 2.0 * x_c * x_c - 1.0
    T3 = 4.0 * x_c * x_c * x_c - 3.0 * x_c
    
    # Fourier series in t (periodic driving)
    St1 = np.sin(t)
    Ct1 = np.cos(t)
    St2 = np.sin(2.0 * t)
    Ct2 = np.cos(2.0 * t)
    
    # Bounded nonlinear damping
    v_tanh = np.tanh(v_c)
    
    # Linear combination
    res = (p0 + 
           p1 * T1 + 
           p2 * T2 + 
           p3 * T3 + 
           p4 * v_c + 
           p5 * v_tanh + 
           p6 * St1 + 
           p7 * Ct1 + 
           p8 * St2 + 
           p9 * Ct2)
    
    return res
