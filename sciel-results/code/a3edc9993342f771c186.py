"""
Strategy: Unconstrained 4th-Order Multivariate Polynomial.

Rationale:
The previous strategy used a multiplicative model (Monod * Gaussian * Quadratic).
This creates a highly non-convex loss landscape for the 10 parameters, causing the
external BFGS optimizer (initialized at all ones) to converge to poor local minima 
(score ~40).

This new approach restructures the model into a linear combination of polynomial basis 
functions:
y = sum(w_i * phi_i(x))

We have 10 parameters. We use them to weight a fixed set of 10 polynomial basis functions
of up to 4th order. The basis functions are constructed from normalized inputs 
(b, s, temp, pH) to ensure they are well-scaled around 0-1.

By making the model linear in the parameters (weights), the loss landscape becomes 
convex (quadratic in weights), allowing BFGS to find the global minimum efficiently 
and reliably, regardless of the initial guess (within reasonable bounds).

Basis Selection:
We select 10 basis functions that span main effects, interactions, and higher-order 
terms to capture non-linearity without assuming specific biological forms like 
symmetric Gaussians or Monod saturation.

Inputs:
b: Population density (range ~[1, 5]) -> normalized to [0, 1]
s: Substrate (range ~[1, 5]) -> normalized to [0, 1]
temp: Temperature (range ~[1, 40]) -> normalized to [0, 1]
pH: pH level (range ~[2, 12]) -> normalized to [0, 1]
"""

# EVOLVE-BLOCK-START
import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a 4th-order multivariate polynomial.
    
    Inputs:
    b: Population density (range ~[1, 5])
    s: Substrate (range ~[1, 5])
    temp: Temperature (range ~[1, 40])
    pH: pH level (range ~[2, 12])
    
    Params (length 10):
    Weights for the 10 polynomial basis functions.
    """
    # Normalize inputs to [0, 1] based on task description ranges
    # This stabilizes the polynomial terms and helps the optimizer.
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 4.0
    t_n = (temp - 1.0) / 39.0
    ph_n = (pH - 2.0) / 10.0
    
    p = params
    
    # Define 10 basis functions.
    # We aim for a mix of:
    # - Constant term (bias)
    # - Main effects (linear)
    # - Interactions (bilinear)
    # - Higher order (quadratic/cubic/quartic) to capture non-monotonicity
    
    # 1. Bias
    phi_0 = np.ones_like(b_n)
    
    # 2-5. Main effects (linear)
    phi_1 = b_n
    phi_2 = s_n
    phi_3 = t_n
    phi_4 = ph_n
    
    # 6-8. Key interactions (bilinear)
    # Substrate and Temperature are often the primary drivers
    phi_5 = s_n * t_n
    # Density and Substrate interaction
    phi_6 = b_n * s_n
    # Temperature and pH interaction
    phi_7 = t_n * ph_n
    
    # 9-10. Higher order terms for non-monotonicity
    # Temperature often has a peak (Gaussian-like), a quadratic term helps
    phi_8 = t_n**2
    # Substrate saturation might be captured by a quadratic or cubic term
    # Using t_n**3 might be too unstable, let's use s_n**2 or a mixed term
    phi_9 = s_n**2
    
    # Combine linearly
    # result = p[0]*phi_0 + ... + p[9]*phi_9
    
    # Construct feature matrix for efficiency (though direct sum is fine for 10 terms)
    # Let's just sum them up to avoid large temporary arrays if possible, 
    # but 7500 rows is small.
    
    result = (
        p[0] * phi_0 +
        p[1] * phi_1 +
        p[2] * phi_2 +
        p[3] * phi_3 +
        p[4] * phi_4 +
        p[5] * phi_5 +
        p[6] * phi_6 +
        p[7] * phi_7 +
        p[8] * phi_8 +
        p[9] * phi_9
    )
    
    return result
# EVOLVE-BLOCK-END
