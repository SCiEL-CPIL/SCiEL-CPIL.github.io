import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Fourier-based basis.
    This structural change leverages the periodic nature of oscillators,
    which polynomials approximate poorly and Tanh approximates only locally.
    """
    # Precompute trigonometric functions for x and v
    # Frequencies 1 and 2 are chosen as a starting point for fundamental and first harmonic
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2 * x)
    cos_2x = np.cos(2 * x)
    
    sin_v = np.sin(v)
    cos_v = np.cos(v)
    
    # Cross terms coupling position and velocity
    # These allow the "restoring force" to depend on velocity (damping/driving)
    sin_x_v = sin_x * v
    cos_x_v = cos_x * v
    x_sin_v = x * sin_v
    
    # Basis terms
    # Centering cosines helps with conditioning (reduces correlation with bias)
    t0 = np.ones_like(x)
    t1 = sin_x
    t2 = cos_x - 1.0
    t3 = sin_2x
    t4 = cos_2x - 1.0
    t5 = sin_v
    t6 = cos_v - 1.0
    t7 = sin_x_v
    t8 = cos_x_v
    t9 = x_sin_v
    
    return (params[0] * t0 +
            params[1] * t1 +
            params[2] * t2 +
            params[3] * t3 +
            params[4] * t4 +
            params[5] * t5 +
            params[6] * t6 +
            params[7] * t7 +
            params[8] * t8 +
            params[9] * t9)
