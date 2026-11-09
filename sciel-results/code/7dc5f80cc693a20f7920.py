"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a Voigt-type saturation model with polynomial corrections.
    """
    # Unpack parameters
    E_base = params[0]
    E_temp = params[1]
    sat_base = params[2]
    sat_temp = params[3]
    n_exp = params[4]
    c1 = params[5]
    c2 = params[6]
    c3 = params[7]
    c4 = params[8]
    offset = params[9]
    
    # Calculate temperature-dependent properties
    E = E_base + E_temp * temp
    sigma_sat = sat_base + sat_temp * temp
    
    # Ensure sigma_sat is positive to avoid division issues
    # If sigma_sat becomes negative during optimization, clip it to a small positive value
    # This helps stability
    sigma_sat_safe = np.maximum(sigma_sat, 1e-6)
    
    # Calculate elastic stress
    x = E * strain
    
    # Calculate saturation term
    # Using absolute value to handle negative strain (compression) smoothly
    abs_x = np.abs(x)
    
    # Voigt model: x / (1 + (x/sat)^n)
    # We apply sign later
    ratio = abs_x / sigma_sat_safe
    
    # Clip ratio to avoid overflow in power calculation if n is large and ratio is large
    # Although inputs are bounded, params might not be
    ratio_clipped = np.clip(ratio, 0, 1e6)
    
    denom = 1.0 + np.power(ratio_clipped, n_exp)
    
    # Safe division
    sat_stress = abs_x / denom
    
    # Apply sign of strain
    sign_strain = np.sign(strain)
    # Handle zero strain case explicitly if needed, but sign(0)=0 works with sat_stress=0
    main_term = sign_strain * sat_stress
    
    # Polynomial corrections
    # These allow the model to deviate from the pure Voigt shape
    poly_term = c1 * strain + c2 * np.power(strain, 2) + c3 * np.power(strain, 3) + c4 * np.power(strain, 4)
    
    return main_term + poly_term + offset

# EVOLVE-BLOCK-END
