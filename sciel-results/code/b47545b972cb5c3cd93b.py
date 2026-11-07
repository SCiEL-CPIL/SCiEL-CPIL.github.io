"""Symbolic regression: stress-strain of an aluminium rod."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod using a Ramberg-Osgood formulation.
    
    The model is structured as:
    sigma = E(T) * eps + K(T) * (eps / eps0)^n
    
    Where:
    - E(T) is the elastic modulus, parameterized as a polynomial in temperature
    - K(T) is the hardening coefficient, parameterized as a polynomial in temperature
    - n is the hardening exponent
    - eps0 is a reference strain for normalization
    
    To keep the model linear in parameters (for BFGS convergence), we expand:
    - E(T) = p0 + p1*T + p2*T^2
    - K(T) = p3 + p4*T + p5*T^2
    - n is fixed at a typical value for aluminium (around 0.1-0.2)
    - eps0 is fixed at a typical yield strain
    
    The term (eps/eps0)^n is nonlinear in eps but since n is fixed, it acts as a basis function.
    We also add interaction terms to capture more complex behavior.
    
    Basis functions (linear in params):
    b0 = 1
    b1 = eps
    b2 = T
    b3 = T*eps
    b4 = T^2
    b5 = T^2*eps
    b6 = (eps/eps0)^n
    b7 = T*(eps/eps0)^n
    b8 = T^2*(eps/eps0)^n
    b9 = eps*(eps/eps0)^n  (additional hardening term)
    
    This structure ensures:
    1. Linear elastic regime: when eps is small, (eps/eps0)^n << eps, so sigma ≈ (p0 + p1*T + p2*T^2)*eps
    2. Plastic hardening: the (eps/eps0)^n terms capture the nonlinear hardening
    3. Temperature dependence: E and K vary with T
    4. Linearity in params: BFGS converges reliably
    """
    s = strain
    t = temp
    p = params
    
    # Fixed physical constants
    # Typical yield strain for aluminium is around 0.001-0.005
    # Using eps0 = 0.002 as reference
    eps0 = 0.002
    
    # Hardening exponent for aluminium is typically 0.1-0.2
    # Using n = 0.15
    n = 0.15
    
    # Compute the nonlinear strain term
    # For negative strain, we need to handle carefully
    # Use absolute value for the power term, but preserve sign for the linear term
    eps_abs = np.abs(s)
    # Safe power: avoid issues with 0^negative or negative bases
    eps_power = np.power(eps_abs, n)
    
    # Basis functions
    b0 = np.ones_like(s)
    b1 = s
    b2 = t
    b3 = s * t
    b4 = t * t
    b5 = s * t * t
    b6 = eps_power
    b7 = t * eps_power
    b8 = t * t * eps_power
    b9 = s * eps_power
    
    # Combine basis functions with parameters
    # p[0], p[1], p[2] -> E(T) = p0 + p1*T + p2*T^2 (elastic modulus)
    # p[3], p[4], p[5] -> K(T) = p3 + p4*T + p5*T^2 (hardening coefficient)
    # p[6], p[7], p[8] -> coefficients for (eps/eps0)^n terms
    # p[9] -> additional hardening term
    
    stress = (p[0] * b1 + p[1] * b3 + p[2] * b5 +
              p[3] * b6 + p[4] * b7 + p[5] * b8 +
              p[6] * b6 + p[7] * b7 + p[8] * b8 +
              p[9] * b9)
    
    # Simplify: combine like terms
    # Actually, let's restructure to be cleaner:
    # E(T)*eps = (p0 + p1*T + p2*T^2) * eps = p0*eps + p1*eps*T + p2*eps*T^2
    # K(T)*(eps/eps0)^n = (p3 + p4*T + p5*T^2) * (eps/eps0)^n
    # Additional term: p9 * eps * (eps/eps0)^n
    
    # Let's redo this more carefully
    # Elastic part: E(T) * eps
    E_T = p[0] + p[1] * t + p[2] * t * t
    elastic = E_T * s
    
    # Hardening part: K(T) * (eps/eps0)^n
    # Note: (eps/eps0)^n = eps^0.15 / eps0^0.15
    # We can absorb eps0^0.15 into K, so just use eps^n
    K_T = p[3] + p[4] * t + p[5] * t * t
    hardening = K_T * eps_power
    
    # Additional term for better fit
    extra = p[6] * s * eps_power + p[7] * t * s * eps_power + p[8] * t * t * s * eps_power
    
    # Offset term
    offset = p[9] * np.ones_like(s)
    
    stress = elastic + hardening + extra + offset
    
    return stress
# EVOLVE-BLOCK-END
