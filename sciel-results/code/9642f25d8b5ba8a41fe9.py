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
    
    The basis functions are chosen to capture the physics of a nonlinear oscillator
    (e.g., Duffing equation) while maintaining the linear-in-parameters property.
    
    Inputs:
        x: displacement
        v: velocity
        params: array of 10 fitted constants [p0, ..., p9]
    
    Model structure:
        p0 * 1
        p1 * x
        p2 * v
        p3 * x^2
        p4 * v^2
        p5 * x*v
        p6 * x^3
        p7 * v^3
        p8 * x^2*v
        p9 * x*v^2
        
    This covers up to 3rd order interactions, sufficient for many nonlinear dynamics,
    and ensures numerical stability for the optimizer.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis terms
    # Note: Using direct powers is efficient and stable for the given input ranges
    # x in [-0.57, 0.53], v in [-0.18, 0.28]
    
    term_0 = np.ones_like(x)
    term_1 = x
    term_2 = v
    term_3 = x * x
    term_4 = v * v
    term_5 = x * v
    term_6 = x * x * x
    term_7 = v * v * v
    term_8 = x * x * v
    term_9 = x * v * v
    
    # Linear combination
    return (
        p0 * term_0 +
        p1 * term_1 +
        p2 * term_2 +
        p3 * term_3 +
        p4 * term_4 +
        p5 * term_5 +
        p6 * term_6 +
        p7 * term_7 +
        p8 * term_8 +
        p9 * term_9
    )

# EVOLVE-BLOCK-END
