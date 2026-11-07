import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled as a rational function
    of position and velocity. This allows for saturation effects and
    complex nonlinear interactions not captured by low-order polynomials.
    """
    # Unpack parameters
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = params
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    
    # Numerator: Odd powers to preserve symmetry of restoring/damping forces
    num = p1 * x + p2 * v + p3 * x3 + p4 * v3 + p5 * x5 + p6 * v5
    
    # Denominator: Even powers + 1 to ensure stability and allow saturation
    # Scaling might be needed if x, v are large, but they are small here.
    den = 1.0 + p7 * x2 + p8 * v2 + p9 * x4 + p10 * v4
    
    # Avoid division by zero (though unlikely with 1.0 offset and small x,v)
    den = np.where(np.abs(den) < 1e-6, 1e-6, den)
    
    return num / den
