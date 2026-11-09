import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator: linear-in-parameters feature model.
    
    Structural change: replaced the parametric oscillator form
    a = -(1+p0*tanh(x))*x - (1+p1*tanh(v))*v + ...
    with a flat linear-in-parameters model using normalized inputs,
    cubic Duffing terms, quadratic damping, and periodic drive.
    
    This gives BFGS a convex optimization landscape (linear in params),
    enabling reliable convergence from all-ones initialization.
    
    Model:
    a = p0 + p1*sin(t) + p2*cos(t) + p3*X + p4*X^2 + p5*X^3
      + p6*V + p7*V^2 + p8*X*V + p9*sin(2*t)
    
    Where X = x/0.2, V = v/0.35 (normalized to ~[-1,1]).
    """
    # Normalize inputs to help BFGS convergence from all-ones
    # x range: [-0.2056, 0.1749] -> scale by 0.2
    # v range: [-0.3391, 0.3624] -> scale by 0.35
    X = x / 0.2
    V = v / 0.35
    
    # Features
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    XV = X * V
    
    # Periodic drive terms
    St = np.sin(t)
    Ct = np.cos(t)
    S2t = np.sin(2.0 * t)
    
    # Linear combination with 10 params
    res = (params[0] * 1.0 +
           params[1] * St +
           params[2] * Ct +
           params[3] * X +
           params[4] * X2 +
           params[5] * X3 +
           params[6] * V +
           params[7] * V2 +
           params[8] * XV +
           params[9] * S2t)
    
    return res
