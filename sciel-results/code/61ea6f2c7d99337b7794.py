"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Polynomial Basis.
    
    Strategy:
    The current approach uses a non-linear basis (Fourier) which creates a non-convex
    optimization landscape, causing BFGS to potentially converge to poor local minima.
    This strategy uses a polynomial basis that is linear in the parameters.
    This ensures the loss function is convex (a quadratic form), allowing BFGS to
    reliably and quickly converge to the global minimum.
    
    We use a complete polynomial basis up to degree 3 in the variables x and v.
    This gives us 10 terms:
    1. 1
    2. x
    3. v
    4. x^2
    5. x*v
    6. v^2
    7. x^3
    8. x^2*v
    9. x*v^2
    10. v^3
    
    Inputs:
        x: displacement
        v: velocity
        params: array of 10 fitted constants [p0, ..., p9]
    
    Model structure:
        p0 + p1*x + p2*v + p3*x^2 + p4*x*v + p5*v^2 + p6*x^3 + p7*x^2*v + p8*x*v^2 + p9*v^3
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute polynomial terms
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    x_v = x * v
    x2_v = x2 * v
    x_v2 = x * v2
    
    return (
        p0 +
        p1 * x +
        p2 * v +
        p3 * x2 +
        p4 * x_v +
        p5 * v2 +
        p6 * x3 +
        p7 * x2_v +
        p8 * x_v2 +
        p9 * v3
    )

# EVOLVE-BLOCK-END
