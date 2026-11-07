import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod using a Voce-type exponential hardening
    formulation with centered strain and a quadratic temperature-dependent correction.
    
    sigma = E(T)*u_c + sigma_y(T) + Q(T)*(1 - exp(-u/H(T))) + (p8 + p9*T)*u_c^2
    
    Where:
      u_c = strain - 0.5  (Centered strain for better conditioning)
      E(T) = p0 + p1*T          (Elastic modulus)
      sigma_y(T) = p2 + p3*T    (Yield stress)
      Q(T) = p4 + p5*T          (Saturation stress range)
      H(T) = softplus(p6) + softplus(p7)*T  (Hardening modulus, constrained positive)
      correction = (p8 + p9*T) * u_c^2      (Quadratic correction for flexibility)
    
    This structure captures the asymptotic saturation of stress at high strains (Voce term)
    while using centered strain to improve optimization stability and a quadratic correction
    to fit nuances that the ideal Voce model might miss.
    
    Parameters (10):
    p0, p1: Elastic modulus coefficients
    p2, p3: Yield stress coefficients
    p4, p5: Saturation stress coefficients
    p6, p7: Hardening modulus coefficients (via softplus to ensure positivity)
    p8, p9: Quadratic correction coefficients
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    u = strain
    T = temp
    
    # Center strain to improve conditioning of polynomial-like terms
    # Range of u is approx [-0.002, 0.964], so center at 0.5
    u_c = u - 0.5
    
    # Helper for stable softplus to avoid overflow/underflow and ensure differentiability
    def stable_softplus(x):
        # For x > 20, softplus(x) ~ x
        # For x <= 20, softplus(x) = log(1 + exp(x))
        return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -50, 50))))
    
    # Temperature-dependent coefficients
    # E(T): Elastic modulus
    E = p0 + p1 * T
    
    # sigma_y(T): Yield stress
    sigma_y = p2 + p3 * T
    
    # Q(T): Saturation stress range
    Q = p4 + p5 * T
    
    # H(T): Hardening modulus (constrained positive via softplus)
    H_const = stable_softplus(p6)
    H_temp_coeff = stable_softplus(p7)
    H = H_const + H_temp_coeff * T
    
    # Ensure H is not too small to avoid numerical issues
    H_safe = np.maximum(H, 1e-6)
    
    # Voce hardening term: Q * (1 - exp(-u/H))
    # Note: We use original u here because the exponential decay is physically tied to absolute strain
    # from the yield point, but for numerical stability in the exponent, we can keep it as is.
    # However, if u is negative, exp(-u/H) > 1, so (1 - exp(...)) is negative.
    # This is physically acceptable for compression if the model allows it, or the optimizer
    # will adjust sigma_y to compensate.
    arg = -u / H_safe
    arg_clipped = np.clip(arg, -50, 50)
    voce_term = Q * (1.0 - np.exp(arg_clipped))
    
    # Elastic contribution (using centered strain for better orthogonality with correction)
    # Wait, physically E*u is the elastic part. If we use E*u_c, we are shifting the zero point.
    # The constant shift is absorbed by sigma_y.
    # Let's stick to E * u_c to help BFGS. The "true" elastic term is E*(u_c + 0.5) = E*u_c + 0.5*E.
    # The 0.5*E term is linear in T and constant in u, so it can be absorbed into sigma_y(T) 
    # if sigma_y had a constant term, but sigma_y is p2+p3*T. 
    # Actually, E*u = E*(u_c + 0.5) = E*u_c + 0.5*E.
    # 0.5*E = 0.5*(p0+p1*T). This is a linear function of T.
    # sigma_y = p2 + p3*T.
    # So the constant offset 0.5*p0 and temp-dependent offset 0.5*p1*T can be absorbed 
    # into p2 and p3 respectively. Thus, using E*u_c is valid and better conditioned.
    elastic = E * u_c
    
    # Quadratic correction term
    # Using centered strain makes this term orthogonal to the linear elastic term in the least squares sense
    # over a symmetric range, which helps BFGS.
    correction = (p8 + p9 * T) * (u_c * u_c)
    
    # Combine all terms
    sigma = elastic + sigma_y + voce_term + correction
    
    return sigma
