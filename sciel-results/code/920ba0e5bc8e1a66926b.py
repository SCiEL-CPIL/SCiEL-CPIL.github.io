import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Model: Physically motivated Ramberg-Osgood style with polynomial corrections.
    sigma = E(T) * strain * (1 + A(T) * (strain / eps_y(T))^n) + quadratic corrections
    """
    # Extract parameters
    E0 = params[0]
    ET = params[1]
    Ay0 = params[2]
    AyT = params[3]
    eps_y0 = params[4]
    eps_yT = params[5]
    n = params[6]
    q0 = params[7]
    qT = params[8]
    qTT = params[9]
    
    # Ensure physical constraints for stability
    # Young's Modulus should be positive
    E = np.maximum(E0 + ET * temp, 1e-6)
    
    # Yield strain scale should be positive
    eps_y = np.maximum(eps_y0 + eps_yT * temp, 1e-6)
    
    # Hardening exponent should be positive
    n_safe = np.maximum(n, 0.1)
    
    # Hardening amplitude
    A = Ay0 + AyT * temp
    
    # Core non-linear term
    # (strain / eps_y) ^ n
    ratio = strain / eps_y
    # Handle negative strain? Strain can be negative in training range [-0.0019, 0.96].
    # If strain is negative, ratio is negative. Negative number to non-integer power is NaN.
    # We need to handle this. 
    # Physically, compression is similar to tension for small strains, but plasticity differs.
    # For simplicity and to avoid NaNs, we can use sign-aware power or just absolute value?
    # Or restrict the power term to only apply for positive strain?
    # Let's use np.sign(strain) * (abs(strain)/eps_y)**n
    power_term = np.sign(strain) * (np.abs(strain) / eps_y) ** n_safe
    
    core_stress = E * strain * (1 + A * power_term)
    
    # Quadratic corrections to improve fit flexibility
    quad_corr = (q0 + qT * temp + qTT * temp**2) * strain**2
    
    stress = core_stress + quad_corr
    
    return stress
