"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series basis with 10 parameters, linear in params.
    
    Domain: x in [0.1, 6].
    Frequencies chosen to cover the domain well.
    
    Model:
    y = p[0] + p[1]*x + p[2]*x^2
        + p[3]*cos(w1*x) + p[4]*sin(w1*x)
        + p[5]*cos(w2*x) + p[6]*sin(w2*x)
        + p[7]*cos(w3*x) + p[8]*sin(w3*x)
        + p[9]*cos(w4*x)
    
    Frequencies:
    w1 = 2*pi / 6 ≈ 1.047
    w2 = 2*pi / 3 ≈ 2.094
    w3 = 2*pi / 2 ≈ 3.142
    w4 = 2*pi / 1.5 ≈ 4.189
    
    All 10 parameters are linear coefficients, making this a linear least-squares
    problem that BFGS can solve very reliably.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure params are finite
    if not np.all(np.isfinite(p)):
        p = np.where(np.isfinite(p), p, 1.0)
    
    # Extract parameters
    c0 = p[0]
    c1 = p[1]
    c2 = p[2]
    c3 = p[3]
    c4 = p[4]
    c5 = p[5]
    c6 = p[6]
    c7 = p[7]
    c8 = p[8]
    c9 = p[9]
    
    # Frequencies
    w1 = 2.0 * np.pi / 6.0
    w2 = 2.0 * np.pi / 3.0
    w3 = 2.0 * np.pi / 2.0
    w4 = 2.0 * np.pi / 1.5
    
    # Compute basis functions
    x2 = x * x
    cos1 = np.cos(w1 * x)
    sin1 = np.sin(w1 * x)
    cos2 = np.cos(w2 * x)
    sin2 = np.sin(w2 * x)
    cos3 = np.cos(w3 * x)
    sin3 = np.sin(w3 * x)
    cos4 = np.cos(w4 * x)
    
    # Combine
    prediction = c0 + c1 * x + c2 * x2 + c3 * cos1 + c4 * sin1 + c5 * cos2 + c6 * sin2 + c7 * cos3 + c8 * sin3 + c9 * cos4
    
    # Check for non-finite values
    if not np.all(np.isfinite(prediction)):
        prediction = np.where(np.isfinite(prediction), prediction, 0.0)
        
    return prediction
# EVOLVE-BLOCK-END
