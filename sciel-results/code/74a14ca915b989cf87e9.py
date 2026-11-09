"""
Symbolic regression for stress-strain of an aluminium rod.
Uses a smooth elastic-plastic model with hyperbolic tangent yield transition
and saturating hardening to capture the continuous stress-strain curve.
The 10 parameters are fitted by the evaluator using BFGS.
"""
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model structure:
    - Elastic response: linear in strain with temperature-dependent modulus
    - Smooth yield transition: tanh-based step function at yield strain
    - Saturating hardening: asymptotic approach to ultimate stress
    - Temperature effects on yield stress, hardening rate, and ultimate stress
    
    Parameters:
    p0: intercept / residual stress
    p1: elastic modulus (Young's modulus)
    p2: yield strain (center of smooth transition)
    p3: yield transition sharpness (inverse width)
    p4: hardening rate (initial plastic slope after yield)
    p5: ultimate stress / saturation level
    p6: temperature scaling for yield strain
    p7: temperature scaling for elastic modulus
    p8: temperature scaling for hardening rate
    p9: temperature scaling for ultimate stress
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Center temperature for better conditioning (range ~0.067 to 1.0)
    temp_c = temp - 0.533
    
    # Temperature-dependent yield strain
    eps_y = p2 + p6 * temp_c
    
    # Temperature-dependent elastic modulus
    E = p1 * (1.0 + p7 * temp_c)
    
    # Temperature-dependent hardening rate
    H = p4 * (1.0 + p8 * temp_c)
    
    # Temperature-dependent ultimate/saturation stress
    sigma_u = p5 * (1.0 + p9 * temp_c)
    
    # Smooth yield transition using tanh
    # This creates a smooth step from elastic to plastic regime
    # tanh(x/2) goes from 0 to 1 as x goes from -inf to +inf
    # We want: 0 below yield, 1 above yield, smooth in between
    x_yield = p3 * (strain - eps_y)
    yield_step = 0.5 * (1.0 + np.tanh(x_yield))
    
    # Elastic stress component
    stress_elastic = E * strain
    
    # Plastic hardening: saturating function
    # After yield, stress increases with hardening but saturates at sigma_u
    # Use: H * (strain - eps_y) * (1 - exp(-k*(strain - eps_y))) or simpler:
    # sigma_u * (1 - exp(-H * max(strain - eps_y, 0)))
    # But to keep it smooth and differentiable, use:
    # plastic = sigma_u * (1 - exp(-H * (strain - eps_y))) for strain > eps_y
    # Smoothed: use the yield_step to blend
    
    # Simple saturating hardening: sigma_u * (1 - exp(-H * (strain - eps_y)))
    # This is always smooth. For strain < eps_y, it gives negative values
    # which we don't want, so multiply by yield_step.
    
    plastic_strain = strain - eps_y
    # Saturating hardening: approaches sigma_u asymptotically
    hardening = sigma_u * (1.0 - np.exp(-H * plastic_strain))
    
    # Only apply hardening after yield (smoothly)
    stress_plastic = yield_step * hardening
    
    # Total stress
    stress = p0 + stress_elastic + stress_plastic
    
    return stress
