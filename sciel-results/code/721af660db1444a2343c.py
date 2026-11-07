"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Extended Ramberg-Osgood model with exponential hardening and
    temperature-dependent coefficients.
    
    stress = E(T)*eps + A(T)*(|eps|+eps0)^n(T)*sign(eps)
             + B(T)*exp(-C(T)*|eps|)*sign(eps)
             + D(T)*eps^2*sign(eps) + offset(T)
    
    Parameters:
    p0, p1: E(T) = p0 + p1*T
    p2, p3: A(T) = p2 + p3*T
    p4, p5: n(T) = p4 + p5*T  (exponent)
    p6:     eps0 offset
    p7, p8: B(T) = p7 + p8*T  (exp hardening coeff)
    p9:     C scale for exp decay
    p10:    D scale for quadratic strain term
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Temperature-dependent elastic modulus
    E = p0 + p1 * temp
    
    # Temperature-dependent power-law hardening coefficient
    A = p2 + p3 * temp
    
    # Temperature-dependent hardening exponent (softplus for smooth positivity)
    n_raw = p4 + p5 * temp
    n = 1.0 + np.log1p(np.exp(n_raw))
    
    # Strain offset to avoid singularity
    eps0 = np.abs(p6) + 1e-6
    abs_strain = np.abs(strain)
    sign_strain = np.sign(strain)
    
    # Power-law (Ramberg-Osgood) hardening term
    hardening_pow = A * (abs_strain + eps0) ** n * sign_strain
    
    # Elastic term
    elastic_term = E * strain
    
    # Exponential hardening/softening term: B(T)*exp(-C*|eps|)
    # This captures rapid initial hardening that saturates
    B = p7 + p8 * temp
    C = np.abs(p9) + 1e-4
    hardening_exp = B * np.exp(-C * abs_strain) * sign_strain
    
    # Quadratic strain correction for better curvature
    quad_term = (p9 * temp) * strain ** 2 * sign_strain
    
    # Total stress
    stress = elastic_term + hardening_pow + hardening_exp + quad_term
    
    return stress
# EVOLVE-BLOCK-END
