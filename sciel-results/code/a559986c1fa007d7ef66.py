"""
Strategy: Adaptive Spline Basis with Learned Knots.

Unlike the current additive polynomial/tanh model which has fixed functional forms,
this approach uses a set of smooth "bump" functions (Gaussian-like kernels) centered
at learnable strain positions (knots). This allows the model to:
1. Place non-linear features exactly where the stress-strain curve changes character
   (e.g., yield point, hardening region).
2. Maintain differentiability everywhere, ensuring BFGS can optimize both the
   amplitudes and the positions of these features.
3. Incorporate temperature as a modulating factor on the amplitude and width of
   these features, capturing thermal softening/hardening effects.

Parameters (10 total):
p[0]: Global offset (mean stress at zero strain/reference temp)
p[1]: Linear strain slope (Young's modulus approx at reference temp)
p[2]: Linear temperature coefficient (thermal expansion/stress offset)
p[3]: Quadratic temperature coefficient
p[4]: Strain-Temp interaction (linear)
p[5]: Knot position 1 (e.g., yield strain)
p[6]: Amplitude 1 (yield strength magnitude)
p[7]: Knot position 2 (e.g., ultimate strain or hardening plateau)
p[8]: Amplitude 2 (hardening/softening magnitude)
p[9]: Width parameter (controls sharpness of transition; larger = sharper)
"""

import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress prediction using an adaptive spline basis.
    """
    # Safety guard: if inputs are not arrays or empty, return zeros
    if len(strain) == 0:
        return np.array([])
    
    p = params
    
    # Extract parameters
    offset = p[0]
    E_ref = p[1]
    T_lin = p[2]
    T_quad = p[3]
    ST_inter = p[4]
    
    knot1 = p[5]
    amp1 = p[6]
    knot2 = p[7]
    amp2 = p[8]
    width = p[9]
    
    # Ensure width is positive and not too small to avoid division by zero/numerical issues
    # Use a softplus-like or absolute value with a minimum floor
    # We want width to control the spread. Let's use exp(width) to ensure positivity and scale.
    # However, if width is very large, exp(width) overflows. 
    # Better: use 1.0 / (1.0 + np.exp(-width)) * some_scale? 
    # Actually, for a Gaussian kernel exp(-x^2 / (2*sigma^2)), sigma should be positive.
    # Let's map p[9] to sigma = 1.0 / (1.0 + np.exp(-np.clip(p[9], -20, 20)))
    # This maps p[9] in (-inf, inf) to sigma in (0, 1).
    # We might want a wider range of sigmas. Let's multiply by 2.0.
    sigma = 2.0 / (1.0 + np.exp(-np.clip(p[9], -20, 20)))
    
    # Temperature centering for stability
    T_c = temp - 0.5
    
    # Base linear elastic response
    # Stress_linear = E(T) * strain
    # E(T) = E_ref * (1 - thermal_softening * T_c) ? 
    # Let's keep it simpler: E(T) = E_ref + T_lin * T_c + T_quad * T_c^2
    # Then Stress = E(T) * strain + offset + other terms
    
    E_T = E_ref + T_lin * T_c + T_quad * T_c**2
    
    # Linear part
    stress_linear = E_T * strain
    
    # Add the adaptive non-linear components
    # These are Gaussian bumps centered at knot1 and knot2 in strain space.
    # The amplitude can be modulated by temperature to capture thermal effects on plasticity.
    
    # Distance in strain space from the knots
    d1 = strain - knot1
    d2 = strain - knot2
    
    # Gaussian kernels
    # exp(-d^2 / (2 * sigma^2))
    # To prevent underflow/overflow, clip the exponent
    exp_arg1 = -0.5 * (d1 / sigma)**2
    exp_arg2 = -0.5 * (d2 / sigma)**2
    
    kernel1 = np.exp(np.clip(exp_arg1, -50, 0))
    kernel2 = np.exp(np.clip(exp_arg2, -50, 0))
    
    # Modulate amplitudes by temperature
    # Simple linear modulation: amp * (1 + k * T_c)
    # We don't have extra params for k, so let's use the interaction param ST_inter for this?
    # Or just keep amplitudes constant and let the linear part handle temp?
    # The linear part handles temp via E_T. 
    # Let's add a temp-dependent shift to the knots? No, that's complex.
    # Let's allow the amplitude to vary with temp using a shared coefficient? 
    # We have 10 params. 
    # Current: offset, E_ref, T_lin, T_quad, ST_inter, k1, a1, k2, a2, sigma
    # We used ST_inter but didn't apply it yet. Let's apply it as a cross-term.
    
    # Interaction term: strain * temp_c * ST_inter
    stress_interaction = ST_inter * strain * T_c
    
    # Total stress
    stress = (offset 
              + stress_linear 
              + stress_interaction 
              + amp1 * kernel1 
              + amp2 * kernel2)
    
    return stress

# EVOLVE-BLOCK-END
