"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a small neural network with tanh activations.
    
    Structure: input -> 4 hidden units (tanh) -> output (linear)
    Parameters: 4 weights + 4 biases for hidden layer (8 params),
                4 output weights + 1 output bias (5 params) = 13 total,
                but we only have 10, so we use a more compact design.
    
    Alternative: Use a mixture of basis functions with non-linear parameterization.
    We'll use: a*x^3 + b*x^2 + c*x + d + e*sin(f*x) + g*cos(h*x) + i*exp(-j*x^2)
    This gives 10 parameters and can capture polynomials, oscillations, and Gaussian features.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Extract parameters
    a, b, c, d = p[0], p[1], p[2], p[3]
    e, f = p[4], p[5]
    g, h = p[6], p[7]
    i, j = p[8], p[9]
    
    # Polynomial part
    poly = a * x**3 + b * x**2 + c * x + d
    
    # Oscillatory part
    osc = e * np.sin(f * x) + g * np.cos(h * x)
    
    # Gaussian part
    gauss = i * np.exp(-j * x * x)
    
    y = poly + osc + gauss
    
    return y
# EVOLVE-BLOCK-END
