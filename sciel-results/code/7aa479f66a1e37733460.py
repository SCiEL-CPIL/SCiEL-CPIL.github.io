import numpy as np
from scipy.optimize import minimize

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a Chebyshev polynomial basis.
    Chebyshev polynomials provide superior approximation properties for bounded
    domains compared to monomials, reducing oscillation artifacts and improving
    BFGS convergence stability.
    
    The 10 parameters correspond to coefficients of a 2D Chebyshev basis:
    - 4 terms in X (position): T0, T1, T2, T3
    - 3 terms in V (velocity): T0, T1, T2
    - 3 cross terms: T1(X)*T1(V), T2(X)*T1(V), T1(X)*T2(V)
    """
    # Scale inputs to [-1, 1] for Chebyshev polynomials
    # x range: [-0.5728, 0.5275] -> center ~ -0.0227, half-width ~ 0.5502
    # v range: [-0.1754, 0.2821] -> center ~ 0.0534, half-width ~ 0.2288
    
    x_center = -0.0227
    x_half = 0.5502
    v_center = 0.0534
    v_half = 0.2288
    
    # Map to [-1, 1]
    X = (x - x_center) / x_half
    V = (v - v_center) / v_half
    
    # Clip to [-1, 1] to handle slight out-of-range values
    X = np.clip(X, -1.0, 1.0)
    V = np.clip(V, -1.0, 1.0)
    
    # Compute Chebyshev polynomials up to order 3
    # T0 = 1, T1 = t, T2 = 2*t^2 - 1, T3 = 4*t^3 - 3*t
    X0 = np.ones_like(X)
    X1 = X
    X2 = 2.0 * X * X - 1.0
    X3 = 4.0 * X * X * X - 3.0 * X
    
    V0 = np.ones_like(V)
    V1 = V
    V2 = 2.0 * V * V - 1.0
    V3 = 4.0 * V * V * V - 3.0 * V
    
    # Build 10-term basis
    # Terms in X: T0(X), T1(X), T2(X), T3(X)
    # Terms in V: T0(V), T1(V), T2(V)
    # Cross terms: T1(X)*T1(V), T2(X)*T1(V), T1(X)*T2(V)
    
    t0 = X0
    t1 = X1
    t2 = X2
    t3 = X3
    t4 = V0
    t5 = V1
    t6 = V2
    t7 = X1 * V1
    t8 = X2 * V1
    t9 = X1 * V2
    
    T = np.column_stack([t0, t1, t2, t3, t4, t5, t6, t7, t8, t9])
    
    return T @ params
