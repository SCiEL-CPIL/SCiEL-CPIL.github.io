"""Symbolic regression: an equation for E. coli growth rate."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a piecewise-linear "growth plateau" structure.
    
    This model uses:
    - A substrate-gated temperature response (Monod-like saturation * polynomial temp).
    - Piecewise-linear clipping for temperature and pH to enforce zero-growth boundaries.
    - A rational inhibition term for density.
    - Interaction terms to capture non-separable effects.
    
    The model is linear in `params`, ensuring the optimization landscape is convex
    for the BFGS solver.
    """
    # 1. Substrate saturation (Monod-like)
    # Using a fixed Ks=1.0 to keep it simple, or we can let it be part of the linear combo
    # Let's stick to the structure of inspiration 3 but with clipping as per directive.
    
    # Monod substrate term
    u_s = s / (1.0 + s)
    
    # 2. Temperature response
    # Piecewise linear with a plateau. 
    # Let's define a "growth window" for temperature.
    # E. coli grows well between ~30C and ~40C.
    # We can use np.clip to create a trapezoidal or triangular response.
    # Let's try a simple linear rise and fall, or just a clipped linear term.
    # Directive: "piecewise-linear 'growth plateau' model using np.clip to enforce strict zero-growth boundaries"
    
    # Let's create a temperature factor that is 0 below 30, 1 between 35-40, 0 above 45.
    # This is a trapezoid.
    # t_low = 30, t_high_start = 35, t_high_end = 40, t_high = 45
    # t_factor = np.clip((temp - 30) / 5, 0, 1) * np.clip((45 - temp) / 5, 0, 1)
    # This is piecewise linear.
    
    t_factor = np.clip((temp - 30.0) / 5.0, 0.0, 1.0) * np.clip((45.0 - temp) / 5.0, 0.0, 1.0)
    
    # 3. pH response
    # E. coli grows well between pH 6.5 and 7.5.
    # p_low = 6.0, p_high_start = 6.5, p_high_end = 7.5, p_high = 8.0
    p_factor = np.clip((pH - 6.0) / 0.5, 0.0, 1.0) * np.clip((8.0 - pH) / 0.5, 0.0, 1.0)
    
    # 4. Density inhibition
    # Rational term: 1 / (1 + b)
    d_factor = 1.0 / (1.0 + b)
    
    # Now combine these with linear params.
    # We have 10 params.
    # p0: bias
    # p1: substrate * temp_factor (main growth term)
    # p2: substrate * p_factor
    # p3: substrate * temp_factor * p_factor (interaction)
    # p4: density inhibition (linear in b, or using d_factor)
    # p5: substrate (linear)
    # p6: temp (linear)
    # p7: pH (linear)
    # p8: substrate * d_factor
    # p9: temp * pH
    
    pred = (
        params[0] * 1.0 +
        params[1] * (u_s * t_factor) +
        params[2] * (u_s * p_factor) +
        params[3] * (u_s * t_factor * p_factor) +
        params[4] * d_factor +
        params[5] * u_s +
        params[6] * temp +
        params[7] * pH +
        params[8] * (u_s * d_factor) +
        params[9] * (temp * pH)
    )
    
    return pred
# EVOLVE-BLOCK-END
