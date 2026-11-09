"""
Strategy: Fixed-Order Polynomial Basis with Temperature-Dependent Coefficients.

The previous approach used learnable Gaussian knots, which creates a non-convex
optimization landscape. BFGS often gets stuck in local minima when optimizing
positions and amplitudes simultaneously.

This strategy uses a fixed polynomial basis in strain (up to degree 4) where the
coefficients are themselves linear functions of temperature.
Stress = Sum_{i=0}^{4} (c_i + d_i * T_norm) * strain^i

Parameters (10 total):
p[0]: c_0 (constant stress term at reference temp)
p[1]: c_1 (linear strain coeff, Young's modulus approx)
p[2]: c_2 (quadratic strain coeff)
p[3]: c_3 (cubic strain coeff)
p[4]: c_4 (quartic strain coeff)
p[5]: d_0 (temp coeff for constant term)
p[6]: d_1 (temp coeff for linear term)
p[7]: d_2 (temp coeff for quadratic term)
p[8]: d_3 (temp coeff for cubic term)
p[9]: d_4 (temp coeff for quartic term)

This model is linear in the parameters (if T is fixed), making the loss surface
convex with respect to the parameters for any fixed T. This allows BFGS to converge
much more reliably and quickly to the optimal set of coefficients, capturing the
non-linear stress-strain behavior (yield, hardening) via the polynomial terms,
and the thermal effects via the temperature-dependent coefficients.
"""

import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress prediction using a polynomial basis in strain with temp-dependent coefficients.
    """
    if len(strain) == 0:
        return np.array([])
    
    p = params
    
    # Normalize temperature to center around 0.5 (range [0.06667, 1])
    # This improves numerical conditioning of the polynomial terms.
    T = temp - 0.5
    
    # Extract coefficients
    c0, c1, c2, c3, c4 = p[0], p[1], p[2], p[3], p[4]
    d0, d1, d2, d3, d4 = p[5], p[6], p[7], p[8], p[9]
    
    # Compute powers of strain
    s = strain
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    
    # Compute effective coefficients for each strain power
    # coeff_i = c_i + d_i * T
    coef0 = c0 + d0 * T
    coef1 = c1 + d1 * T
    coef2 = c2 + d2 * T
    coef3 = c3 + d3 * T
    coef4 = c4 + d4 * T
    
    # Construct the polynomial
    # Stress = coef0 + coef1*s + coef2*s^2 + coef3*s^3 + coef4*s^4
    stress = coef0 + coef1 * s + coef2 * s2 + coef3 * s3 + coef4 * s4
    
    return stress

# EVOLVE-BLOCK-END
