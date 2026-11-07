import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute terms
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    x2v = x2 * v
    xv2 = x * v2
    
    # Damped nonlinear oscillator:
    # - p0*x: linear restoring force
    # - p1*v: damping
    # - p2: constant driving force
    # - p3*x2: quadratic nonlinearity
    # - p4*x3: cubic nonlinearity (Duffing-like)
    # - p5*xv: velocity-dependent damping (nonlinear damping)
    # - p6*x2v: higher-order damping
    # - p7*v2: velocity squared term
    # - p8*xv2: mixed term
    # - p9: constant offset
    
    return (p0*x + p1*v + p2 + p3*x2 + p4*x3 + 
            p5*xv + p6*x2v + p7*v2 + p8*xv2 + p9)
