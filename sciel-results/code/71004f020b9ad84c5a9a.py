import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod using a Ramberg-Osgood formulation.
    
    sigma = E(T)*u + K(T) * (u)^n(T) + c(T)*u^2 + d(T)*u^3
    
    Where:
      E(T) = p0 + p1*T          (Elastic modulus)
      K(T) = p2 + p3*T          (Hardening coefficient)
      n(T) = softplus(p4) + softplus(p5)*T  (Hardening exponent, constrained positive)
      c(T) = p6 + p7*T          (Quadratic correction)
      d(T) = p8 + p9*T          (Cubic correction)
    
    The Ramberg-Osgood model captures the transition from elastic to plastic behavior
    through the power-law term K*u^n, where n < 1 gives a smooth yield point transition.
    Additional polynomial terms provide flexibility for fitting nuances.
    
    Parameters (10):
    p0, p1: Elastic modulus coefficients
    p2, p3: Hardening coefficient coefficients
    p4, p5: Hardening exponent coefficients (via softplus to ensure positivity)
    p6, p7: Quadratic correction coefficients
    p8, p9: Cubic correction coefficients
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    u = strain
    T = temp
    
    # Helper for stable softplus to avoid overflow/underflow and ensure differentiability
    def stable_softplus(x):
        return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -50, 50))))
    
    # Temperature-dependent coefficients
    E = p0 + p1 * T
    K = p2 + p3 * T
    n_const = stable_softplus(p4)
    n_temp_coeff = stable_softplus(p5)
    n = n_const + n_temp_coeff * T
    
    c = p6 + p7 * T
    d = p8 + p9 * T
    
    # Ensure n is positive and not too extreme
    n_safe = np.clip(n, 0.1, 10.0)
    
    # Ramberg-Osgood hardening term: K * u^n
    # Use absolute value for negative strains to handle compression
    u_abs = np.abs(u)
    # Clip to avoid 0^negative or very large values
    u_clipped = np.clip(u_abs, 1e-10, 10.0)
    roce_term = K * np.power(u_clipped, n_safe)
    # Preserve sign for negative strains
    roce_term = np.sign(u) * roce_term
    
    # Elastic term
    elastic = E * u
    
    # Polynomial corrections
    quad = c * u * u
    cubic = d * u * u * u
    
    # Combine all terms
    sigma = elastic + roce_term + quad + cubic
    
    return sigma
