import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Center and scale inputs to [-1, 1] to improve conditioning
    # Domain [0.5, 2] -> Center 1.25, Range 1.5 -> Scale 0.75
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75

    p = params
    
    # Compute basis functions
    # A(u) = p0 + p1*u + p2*u^2
    A = p[0] + p[1] * u + p[2] * u**2
    # B(v) = p3 + p4*v + p5*v^2
    B = p[3] + p[4] * v + p[5] * v**2
    # C(w) = p6 + p7*w + p8*w^2
    C = p[6] + p[7] * w + p[8] * w**2
    
    # Product + Bias
    return A * B * C + p[9]
