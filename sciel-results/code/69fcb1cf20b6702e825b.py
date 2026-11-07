"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Ramberg-Osgood based stress-strain model for aluminium.
    
    sigma = E(T) * strain + K(T) * |strain|^n * sign(strain) + C(T)
    
    where E(T), K(T), C(T) are linear in temperature, and n is a global exponent.
    
    params (length 10):
      p0: base elastic modulus E0
      p1: temp coefficient for E: dE/dT
      p2: base hardening coefficient K0
      p3: temp coefficient for K: dK/dT
      p4: base offset C0
      p5: temp coefficient for C: dC/dT
      p6: exponent n (nonlinearity)
      p7: interaction term coefficient for strain*temp
      p8: quadratic strain coefficient (small correction)
      p9: quadratic temp coefficient (small correction)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent coefficients
    E = p0 + p1 * temp
    K = p2 + p3 * temp
    C = p4 + p5 * temp
    
    # Handle sign of strain for the hardening term
    sign_strain = np.sign(strain)
    abs_strain = np.abs(strain)
    
    # Ramberg-Osgood: elastic + plastic hardening
    # sigma = E*eps + K*|eps|^n * sign(eps) + C
    # Clamp n to a reasonable range to avoid numerical issues
    n = np.clip(p6, 1.01, 5.0)
    
    hardening = K * (abs_strain ** n) * sign_strain
    
    # Additional correction terms
    interaction = p7 * strain * temp
    quad_strain = p8 * strain ** 2
    quad_temp = p9 * temp ** 2
    
    stress = E * strain + hardening + C + interaction + quad_strain + quad_temp
    
    return stress
# EVOLVE-BLOCK-END
