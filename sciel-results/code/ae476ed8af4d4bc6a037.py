"""E. coli growth rate model with enhanced polynomial and biological basis functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled as a linear combination of basis functions
    combining Monod kinetics, Gaussian environmental responses, and polynomial
    interaction terms. The 10 parameters are linear coefficients, making this
    a convex least-squares problem that BFGS can solve reliably.
    
    Key improvements over previous versions:
    - Added quadratic terms for substrate and temperature
    - Added pH-squared term for asymmetric pH response
    - Added substrate-temperature-pH three-way interaction
    - Improved density inhibition with substrate coupling
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Fixed biological constants (based on E. coli physiology)
    Ks = 5.0        # Substrate half-saturation constant
    T_opt = 37.0    # Optimal temperature (Celsius)
    T_sig = 5.0     # Temperature sensitivity (width)
    pH_opt = 7.0    # Optimal pH
    pH_sig = 1.0    # pH sensitivity (width)
    Kd = 1.0        # Density inhibition constant
    
    # Core basis functions
    # Monod substrate saturation (captures diminishing returns)
    monod_s = s / (Ks + s)
    
    # Gaussian temperature response (optimal range)
    gauss_temp = np.exp(-((temp - T_opt) ** 2) / (2.0 * T_sig ** 2))
    
    # Gaussian pH response (optimal range)
    gauss_ph = np.exp(-((pH - pH_opt) ** 2) / (2.0 * pH_sig ** 2))
    
    # Density inhibition (crowding effect)
    density_inhib = 1.0 / (1.0 + Kd * b)
    
    # Enhanced basis functions with polynomial terms
    # Substrate quadratic term (captures non-linear saturation)
    s_quad = s ** 2
    
    # Temperature quadratic (captures asymmetric response)
    temp_quad = (temp - T_opt) ** 2
    
    # pH quadratic (captures asymmetric pH response)
    ph_quad = (pH - pH_opt) ** 2
    
    # Key interaction terms
    temp_ph_interaction = gauss_temp * gauss_ph
    substrate_temp_interaction = monod_s * gauss_temp
    substrate_ph_interaction = monod_s * gauss_ph
    density_substrate_interaction = density_inhib * monod_s
    
    # Three-way interaction: substrate * temp * pH
    triple_interaction = monod_s * gauss_temp * gauss_ph
    
    # Combined basis functions
    # 1. Full interaction: substrate * temp * pH * density
    basis_1 = monod_s * gauss_temp * gauss_ph * density_inhib
    
    # 2. Substrate * temp
    basis_2 = monod_s * gauss_temp
    
    # 3. Substrate * pH
    basis_3 = monod_s * gauss_ph
    
    # 4. Temp * pH
    basis_4 = gauss_temp * gauss_ph
    
    # 5. Substrate only
    basis_5 = monod_s
    
    # 6. Temperature only (with quadratic correction)
    basis_6 = gauss_temp * (1.0 - 0.1 * temp_quad / (T_sig ** 2))
    
    # 7. pH only (with quadratic correction for asymmetry)
    basis_7 = gauss_ph * (1.0 - 0.1 * ph_quad / (pH_sig ** 2))
    
    # 8. Density * substrate
    basis_8 = density_inhib * monod_s
    
    # 9. Three-way interaction (substrate * temp * pH)
    basis_9 = triple_interaction
    
    # 10. Constant (baseline growth)
    basis_10 = np.ones_like(b)
    
    # Combine all basis functions with fitted parameters
    prediction = (
        p0 * basis_1 +
        p1 * basis_2 +
        p2 * basis_3 +
        p3 * basis_4 +
        p4 * basis_5 +
        p5 * basis_6 +
        p6 * basis_7 +
        p7 * basis_8 +
        p8 * basis_9 +
        p9 * basis_10
    )
    
    return prediction
# EVOLVE-BLOCK-END
