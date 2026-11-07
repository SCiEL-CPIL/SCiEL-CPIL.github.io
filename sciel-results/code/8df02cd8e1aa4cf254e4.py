"""
Strategy: Flexible Spline-on-Strain Model.
Instead of enforcing a specific physics equation (like Ramberg-Osgood), 
we use a set of smooth, non-linear basis functions of strain to capture 
the general shape of the stress-strain curve. The parameters control the 
amplitude, curvature, and temperature-dependence of these basis functions.

This approach is structurally different from the current program:
1. It does not assume a specific yield point or hardening exponent form.
2. It uses a sum of basis functions (sigmoid-like and polynomial-like) 
   which can be optimized more robustly by BFGS than a single power law 
   with complex temperature couplings.
3. It handles negative and positive strain smoothly without explicit 
   sign-handling logic for power laws.

Basis Functions:
- b1: Linear elastic term (strain)
- b2: Smooth transition term (tanh-like) to capture yield
- b3: Hardening term (quadratic/cubic mix)
- Temperature modulates the coefficients of these terms.
"""

import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Inputs:
    strain: array of strain values
    temp: array of temperature values
    params: array of 10 parameters
    
    Returns:
    array of stress values
    """
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    
    # Normalize temperature to [0, 1] range approximately for stability
    # Training range is [0.06667, 1]
    T_norm = (T - 0.06667) / (1.0 - 0.06667)
    T_norm = np.clip(T_norm, 0.0, 1.0)
    
    p = params
    
    # Extract parameters
    # We use a mix of direct and transformed parameters to ensure stability
    # p[0]: Scale for linear term
    # p[1]: Scale for yield transition term
    # p[2]: Scale for hardening term
    # p[3]: Temperature effect on linear term
    # p[4]: Temperature effect on yield term
    # p[5]: Temperature effect on hardening term
    # p[6]: Shift for yield transition (strain offset)
    # p[7]: Width/steepness of yield transition
    # p[8]: Global scale
    # p[9]: Global offset
    
    # To ensure the "width" parameter p[7] is positive for the tanh argument,
    # we use exp(p[7]). This ensures the transition is always smooth and defined.
    width = np.exp(p[7])
    
    # Calculate basis functions
    # 1. Linear term: E * eps
    b1 = p[0] * eps
    
    # 2. Yield transition term: 
    # Uses a tanh-like function to model the sharp drop in slope at yield.
    # The argument is scaled by width.
    # We want this to be 0 at low strain and saturate at high strain.
    # tanh(x) goes from -1 to 1. We shift and scale it.
    # Let's use a sigmoid-like shape: 1 / (1 + exp(-k*(x-x0)))
    # But tanh is often better for symmetry if we had negative yield, 
    # but here we can use a simple smooth step.
    # Let's use: p[1] * tanh(p[7] * (eps - p[6]))
    # This creates a smooth S-curve centered at p[6].
    b2 = p[1] * np.tanh(width * (eps - p[6]))
    
    # 3. Hardening term:
    # A polynomial term that grows with strain to capture strain hardening.
    # We use a squared term to ensure it's positive and smooth.
    # To avoid large values for negative strain if p[2] is negative, 
    # we can just let the optimizer find the sign.
    # b3 = p[2] * (eps ** 2)
    # Let's make it slightly more flexible: p[2] * (eps ** 2) * exp(-p[8]*eps) ? No, keep it simple.
    # A pure quadratic might not be enough for large strains if the curve is concave down.
    # Let's use a term that saturates: p[2] * eps / (1 + eps/alpha)
    # But we only have 10 params.
    # Let's stick to a polynomial mix: p[2] * eps**2
    b3 = p[2] * (eps ** 2)
    
    # Temperature modulation
    # The physical properties of aluminium change with temperature.
    # Typically, E and Yield Strength decrease with T.
    # We modulate the coefficients of the basis functions with T_norm.
    
    coeff1 = 1.0 + p[3] * T_norm
    coeff2 = 1.0 + p[4] * T_norm
    coeff3 = 1.0 + p[5] * T_norm
    
    # Combine basis functions with temperature-modulated coefficients
    sigma = (b1 * coeff1 + b2 * coeff2 + b3 * coeff3)
    
    # Global scale and offset
    sigma = p[8] * sigma + p[9]
    
    # Ensure finite output
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    return sigma
