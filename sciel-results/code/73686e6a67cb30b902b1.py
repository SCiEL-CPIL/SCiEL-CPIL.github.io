import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model using Ramberg-Osgood formulation.
    
    sigma = E(T) * epsilon + K(T) * (epsilon * n(T))^n(T)
    
    Where E, K, n are parameterized as low-order polynomials in temperature.
    This captures the linear elastic regime and nonlinear hardening behavior.
    
    Parameters (10):
    p0, p1, p2: E(T) = p0 + p1*T + p2*T^2 (elastic modulus)
    p3, p4, p5: K(T) = p3 + p4*T + p5*T^2 (hardening coefficient)
    p6, p7: n(T) = p6 + p7*T (hardening exponent)
    p8, p9: Additional strain correction terms
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    u = strain
    T = temp
    
    # Elastic modulus as quadratic in temperature
    E = p0 + p1 * T + p2 * T * T
    
    # Hardening coefficient as quadratic in temperature
    K = p3 + p4 * T + p5 * T * T
    
    # Hardening exponent as linear in temperature
    n = p6 + p7 * T
    
    # Ensure n stays in a reasonable range for power law (typically 1 < n < 10 for metals)
    # Clip to avoid numerical issues with very small or negative exponents
    n_safe = np.clip(n, 0.5, 20.0)
    
    # Ramberg-Osgood: sigma = E*epsilon + K*(epsilon*n)^n
    # For small epsilon, the second term dominates the nonlinear behavior
    # Use absolute value for the power term to handle potential negative strains
    eps_abs = np.abs(u)
    
    # The power law term: K * (eps_abs * n_safe)^n_safe
    # This gives smooth yield transition
    power_term = K * np.power(eps_abs * n_safe + 1e-12, n_safe)
    
    # Combine elastic and plastic components
    # The sign should match the strain
    sigma = E * u + np.sign(u) * power_term
    
    # Add small correction terms for better fit
    # These capture any deviations from the basic Ramberg-Osgood form
    sigma = sigma + p8 * u * (1 - u) + p9 * u * T
    
    return sigma
