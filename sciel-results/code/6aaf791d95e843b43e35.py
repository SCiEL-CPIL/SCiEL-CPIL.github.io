import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod using Ramberg-Osgood formulation.
    
    sigma = E(T) * epsilon + K(T) * |epsilon|^n(T) * sign(epsilon)
    
    Where:
    E(T) = p0 + p1*T + p2*T^2   (elastic modulus, temperature-dependent)
    K(T) = p3 + p4*T + p5*T^2   (hardening coefficient, temperature-dependent)
    n(T) = p6 + p7*T            (hardening exponent, temperature-dependent)
    p8, p9: additional correction terms for higher-order behavior
    
    Parameters (10):
    p0, p1, p2: Elastic modulus as quadratic in temperature
    p3, p4, p5: Hardening coefficient as quadratic in temperature
    p6, p7: Hardening exponent as linear in temperature
    p8, p9: Additional correction terms
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale parameters to help BFGS converge from all-ones initialization
    # Elastic modulus E ~ 70 GPa for aluminium, strain ~ 1, so sigma ~ 70
    # We need E to be large relative to 1, so scale p0-p2 by ~10
    # K should be similar magnitude, scale p3-p5 by ~10
    # n should be ~ 2-5 for metals, scale p6-p7 by ~1
    # p8, p9 are small corrections, scale by ~1
    
    E = (p0 + p1 * temp + p2 * temp**2) * 10.0
    K = (p3 + p4 * temp + p5 * temp**2) * 10.0
    n = (p6 + p7 * temp) + 1.0  # offset to ensure n > 1
    
    u = strain
    T = temp
    
    # Ramberg-Osgood: sigma = E*eps + K*|eps|^n*sign(eps)
    # For negative strains, we use sign(eps) to maintain odd symmetry
    # |u|^n for n possibly non-integer: use exp(n*log(|u|)) for u != 0
    abs_u = np.abs(u)
    sign_u = np.sign(u)
    
    # Handle u=0 case safely
    eps_safe = np.where(abs_u > 1e-15, abs_u, 1.0)
    hardening = K * np.power(eps_safe, n)
    hardening = np.where(abs_u > 1e-15, hardening, 0.0)
    
    sigma = E * u + sign_u * hardening
    
    # Add small correction terms for flexibility
    # These capture deviations from pure Ramberg-Osgood
    corr1 = p8 * u**2 * (1.0 - T)  # quadratic correction, weaker at high T
    corr2 = p9 * u**3 * T          # cubic correction, stronger at high T
    
    sigma = sigma + corr1 + corr2
    
    return sigma
