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
    reliably and quickly converge to the global optimum.
    
    We select a set of monomials up to degree 3 that capture the likely physics
    of a nonlinear oscillator (e.g., Duffing-like terms x^3, v^3, x*v).
    
    Inputs:
        x: displacement
        v: velocity
        params: array of 10 fitted constants [p0, ..., p9]
    
    Model structure:
        p0 * 1          (bias)
        p1 * x          (linear restoring/damping)
        p2 * v          
        p3 * x^2        (nonlinear stiffness/drag)
        p4 * v^2        
        p5 * x*v        (coupling)
        p6 * x^3        (strong nonlinear restoring)
        p7 * v^3        (strong nonlinear damping)
        p8 * x^2*v      (higher order coupling)
        p9 * x*v^2      (higher order coupling)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers to avoid redundant calculations
    x2 = x * x
    v2 = v * v
    x3 = x2 * x
    v3 = v2 * v
    
    return (
        p0 +
        p1 * x +
        p2 * v +
        p3 * x2 +
        p4 * v2 +
        p5 * x * v +
        p6 * x3 +
        p7 * v3 +
        p8 * x2 * v +
        p9 * x * v2
    )

# EVOLVE-BLOCK-END
