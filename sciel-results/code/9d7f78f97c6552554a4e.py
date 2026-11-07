import numpy as np

def equation(strain, temp, params):
    """
    Symbolic regression for stress-strain of an aluminium rod.
    
    Strategy: Physically-Constrained Ramberg-Osgood Model.
    
    This strategy moves away from generic polynomial basis expansions (which can suffer 
    from Runge's phenomenon and poor extrapolation/interpolation balance) to a model 
    explicitly derived from the Ramberg-Osgood constitutive law, adapted for temperature 
    dependence.
    
    The Ramberg-Osgood law describes the total strain as the sum of elastic and plastic 
    components:
    epsilon = sigma/E + (sigma/K)^n
    
    Rearranging to solve for stress (sigma) is implicit, but for symbolic regression where 
    we must provide a closed-form sigma(epsilon, T), we can use an explicit approximation 
    or a basis that mimics this structure. A common explicit form that captures the 
    "kink" and hardening is:
    
    sigma = E(T) * epsilon + H(T) * (epsilon - epsilon_y(T))^p
    
    However, to maintain differentiability and smoothness for BFGS, and to allow the 
    optimizer to find the best fit without strict physical constraints on the yield point 
    (which might be noisy or vary), we use a smooth approximation of the plastic term.
    
    We define the stress as a linear combination of basis functions that reflect this physics:
    1. Elastic term: E * strain
    2. Elastic temp correction: E_t * strain * temp
    3. Plastic hardening term: (max(strain - s_y, 0))^p
       To make this smooth and differentiable everywhere, we use a softplus-like function 
       or simply a power of a shifted strain with a small epsilon to avoid 0^0 issues, 
       but since BFGS is sensitive, a pure polynomial is safer. 
       
    Actually, the "Linear-in-Parameters" approach in the current program is already 
    linear. The key insight from the inspirations is that **fractional powers** (Ramberg-Osgood) 
    are more physically accurate than integer polynomials for hardening.
    
    Let's use a model that is linear in parameters but uses fractional powers in the basis.
    This is still linear in parameters, so BFGS (or any linear solver) will find the global 
    minimum for the fixed basis. The improvement comes from the *choice of basis*.
    
    Basis Functions:
    phi_1 = strain
    phi_2 = strain * temp
    phi_3 = (strain - 0.002)_+^0.5  (Square root hardening, common in some metals)
    phi_4 = (strain - 0.002)_+^0.5 * temp
    phi_5 = (strain - 0.002)_+^1.5  (Cubic-root-like or higher order)
    phi_6 = (strain - 0.002)_+^1.5 * temp
    phi_7 = (strain - 0.002)_+^2.0
    phi_8 = (strain - 0.002)_+^2.0 * temp
    phi_9 = 1.0
    phi_10 = temp
    
    Wait, the prompt says "up to 10 fitted constants". The evaluator fits `params`.
    If I use `np.maximum(strain - 0.002, 0)`, it is non-differentiable at the kink.
    BFGS uses gradients. If the gradient is undefined, BFGS might fail or get stuck.
    To be safe, I should use a smooth approximation for the positive part.
    
    Smooth positive part: x_+ = 0.5 * (x + sqrt(x^2 + eps))
    Let x = strain - s_y, where s_y is a fixed small value, e.g., 0.002.
    eps = 1e-6 for numerical stability.
    
    Basis:
    1. strain
    2. strain * temp
    3. (x_+)^0.5
    4. (x_+)^0.5 * temp
    5. (x_+)^1.0
    6. (x_+)^1.0 * temp
    7. (x_+)^1.5
    8. (x_+)^1.5 * temp
    9. 1.0
    10. temp
    
    This basis is smooth (C^1 at least if eps is small enough, actually C^infty if we consider the sqrt term carefully, but the derivative of sqrt(x^2+eps) is x/sqrt(x^2+eps) which is 0 at x=0, so the derivative of the whole softplus-like term is continuous).
    
    Why this is different from the current program:
    Current: Integer powers of strain (s, s^2, s^3, s^4).
    New: Fractional powers of a *shifted* strain (hardening only after yield).
    
    This physically separates the elastic region (linear in strain) from the plastic region 
    (fractional power in strain - yield). This should allow the model to fit the elastic 
    slope accurately without the plastic terms interfering, and fit the hardening curve 
    accurately without the elastic terms interfering.
    """
    s = strain
    t = temp
    p = params
    
    # Fixed yield strain estimate for aluminium (typical range 0.001 - 0.005)
    s_y = 0.002
    
    # Smooth positive part approximation
    # x_+ approx 0.5 * (x + sqrt(x^2 + eps))
    x = s - s_y
    eps = 1e-8
    x_smooth = 0.5 * (x + np.sqrt(x * x + eps))
    
    # Ensure non-negative (the softplus approximation is always > 0, but we want it to be ~0 for x < 0)
    # Actually, 0.5*(x + sqrt(x^2+eps)) is always positive. For x << 0, it is approx 0.5*(x - |x|) = 0.
    # So it is a good smooth approximation of max(x, 0).
    
    # Calculate fractional powers
    # Note: x_smooth is always >= 0, so fractional powers are well-defined.
    x05 = np.sqrt(x_smooth)       # Power 0.5
    x10 = x_smooth                # Power 1.0
    x15 = x_smooth * x05          # Power 1.5
    x20 = x_smooth * x_smooth     # Power 2.0
    
    # Basis functions
    # 0: Elastic
    b0 = s
    # 1: Elastic temp
    b1 = s * t
    # 2: Plastic 0.5
    b2 = x05
    # 3: Plastic 0.5 temp
    b3 = x05 * t
    # 4: Plastic 1.0
    b4 = x10
    # 5: Plastic 1.0 temp
    b5 = x10 * t
    # 6: Plastic 1.5
    b6 = x15
    # 7: Plastic 1.5 temp
    b7 = x15 * t
    # 8: Intercept
    b8 = 1.0
    # 9: Temp offset
    b9 = t
    
    # Linear combination
    stress = (p[0] * b0 + 
              p[1] * b1 + 
              p[2] * b2 + 
              p[3] * b3 + 
              p[4] * b4 + 
              p[5] * b5 + 
              p[6] * b6 + 
              p[7] * b7 + 
              p[8] * b8 + 
              p[9] * b9)
    
    return stress
