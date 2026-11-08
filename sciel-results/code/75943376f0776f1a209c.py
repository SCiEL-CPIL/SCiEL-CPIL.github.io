import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress model with rational plastic hardening and polynomial correction.
    
    sigma = E(T)*eps + (K1(T)*eps + K2(T)*eps^2) / (1 + H1(T)*eps) + C(T)*eps^3
    
    Params:
    p[0]: E0
    p[1]: c_E
    p[2]: K1_0
    p[3]: c_K1
    p[4]: K2_0
    p[5]: c_K2
    p[6]: h1_0 (log H1_0)
    p[7]: c_h1
    p[8]: C_0
    p[9]: c_C
    """
    eps = strain
    T = temp
    
    E0 = params[0]
    c_E = params[1]
    
    K1_0 = params[2]
    c_K1 = params[3]
    
    K2_0 = params[4]
    c_K2 = params[5]
    
    h1_0 = params[6]
    c_h1 = params[7]
    
    C_0 = params[8]
    c_C = params[9]
    
    # Temp dependent coeffs
    E_T = E0 * (1.0 + c_E * T)
    
    K1_T = K1_0 * (1.0 + c_K1 * T)
    K2_T = K2_0 * (1.0 + c_K2 * T)
    
    # Ensure H1 is positive to avoid singularities in denominator for negative eps
    # H1 = exp(h1)
    h1_T = h1_0 + c_h1 * T
    # Clip h1_T to prevent overflow in exp
    h1_T = np.clip(h1_T, -10, 10)
    H1_T = np.exp(h1_T)
    
    C_T = C_0 * (1.0 + c_C * T)
    
    # Elastic term
    elastic = E_T * eps
    
    # Plastic rational term
    # Denominator: 1 + H1_T * eps
    # For eps > 0, denom > 1.
    # For eps < 0, denom = 1 - H1_T * |eps|. 
    # Since |eps| is small for negative values, this is safe unless H1_T is huge.
    # With h1 clipped to 10, H1_T ~ 22000. 
    # Min denom ~ 1 - 22000 * 0.002 = 1 - 44 = -43. 
    # This crosses zero! 
    # We need to handle this. 
    # If we use H1_T * eps^2 in denom? No, loses linear behavior.
    # If we use 1 + H1_T * (eps/10)? No.
    
    # Let's use a safer rational form:
    # denom = 1 + H1_T * eps + H2_T * eps^2 ?
    # Or just use a polynomial if rational is risky.
    
    # Let's revert to the 1.82 model structure but with better conditioning?
    # The 1.82 model was:
    # sigma = E(T) * eps * (1 + A(T) + B(T)eps + C(T)eps^2 + D(T)eps^3)
    # This is a 4th order polynomial.
    
    # Why did the tanh model (4.76) fail?
    # Maybe the optimizer got stuck.
    
    # Let's try a model that is a polynomial but with a "soft" saturation term added?
    # sigma = E(T) * eps + K(T) * (1 - exp(-alpha(T) * eps))
    # This is safe for eps > 0. For eps < 0, it grows exponentially.
    # But eps < 0 is small.
    
    # Let's try the polynomial model again, but maybe with a 5th order term?
    # Or just the 1.82 model is the best base.
    
    # Let's try to improve the 1.82 model by adding a temperature-dependent non-linearity 
    # that isn't just a linear temp coeff.
    
    # Actually, the 1.82 model uses 10 params:
    # E0, c_E
    # A0, A1
    # B0, B1
    # C0, C1
    # D0, D1
    
    # What if we use a 5th order term and drop the linear A(T) term?
    # sigma = E(T) * eps * (1 + B(T)eps + C(T)eps^2 + D(T)eps^3 + E(T)eps^4)
    # Same number of params.
    
    # Let's try a different functional form: 
    # sigma = E(T) * eps + K(T) * eps^2 / (1 + H(T) * eps)
    # To avoid the zero-crossing issue for negative eps, we can use:
    # denom = 1 + H(T) * (eps + offset) ?
    
    # Let's stick to the polynomial but ensure it's well-conditioned.
    # The inputs are small. 
    # Maybe the issue is that BFGS from ones is bad for high-order polynomials?
    
    # Let's try a model with fewer high-order terms and a saturation.
    # sigma = E(T) * eps + K(T) * tanh(alpha(T) * eps) * eps
    # This was similar to the 4.76 model.
    
    # Let's try:
    # sigma = E(T) * eps + K(T) * (1 - exp(-alpha(T) * eps))
    # For eps < 0, this is fine (negative stress).
    # Params:
    # E0, c_E
    # K0, c_K
    # alpha0, c_alpha
    # Total 6.
    # Add quadratic correction: C(T) * eps^2
    # C0, c_C
    # Total 8.
    # Add cubic correction: D(T) * eps^3
    # D0, c_D
    # Total 10.
    
    # This model:
    # sigma = E(T)*eps + K(T)*(1 - exp(-alpha(T)*eps)) + C(T)*eps^2 + D(T)*eps^3
    
    # This is robust. exp is stable.
    # For large positive eps, exp(-alpha*eps) -> 0, so sigma ~ E*eps + K + C*eps^2 + D*eps^3.
    # This allows for continued hardening if C, D are positive.
    # If we want saturation, K dominates.
    
    # Let's implement this.
    
    E_T = E0 * (1.0 + c_E * T)
    K_T = K0 * (1.0 + c_K * T)
    alpha_T = alpha0 * (1.0 + c_alpha * T)
    C_T = C0 * (1.0 + c_C * T)
    D_T = D0 * (1.0 + c_D * T)
    
    # Clip alpha_T * eps to prevent overflow in exp
    # alpha_T * eps can be large.
    # exp(-large) -> 0.
    # exp(large) -> inf.
    # We have 1 - exp(-x).
    # If x is very negative (eps negative, alpha positive), -x is positive large.
    # exp(large) is inf.
    # So we must clip the argument of exp.
    
    arg = -alpha_T * eps
    # Clip arg to [-50, 50]
    arg = np.clip(arg, -50, 50)
    
    plastic = K_T * (1.0 - np.exp(arg))
    
    stress = E_T * eps + plastic + C_T * (eps * eps) + D_T * (eps * eps * eps)
    
    return stress
