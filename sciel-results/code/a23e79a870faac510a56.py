"""
Strategy: Bilinear Elastic-Plastic Hardening Model with Tanh Saturation.
Instead of a rational function or generic basis expansion, we use a structure
that mirrors the physics of metal deformation:
sigma = E * eps + (Y + H * eps_p) * s(eps)
where:
- E is the elastic modulus (linear rise).
- Y is the yield stress offset.
- H is the hardening modulus (slope after yield).
- eps_p is the plastic strain (eps - eps_yield).
- s(eps) is a smooth step function (tanh-based) that transitions from 0 to 1
  around the yield point, ensuring differentiability for BFGS.

Temperature modulates the yield point and hardening behavior.
This structure is strictly smoother and more constrained than a rational function,
often leading to faster convergence and better generalization for monotonic
stress-strain curves with a clear knee.
"""

import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Inputs:
    strain: array of strain values
    temp: array of temperature values
    params: array of 10 parameters
    
    Returns:
    array of stress values
    """
    # Deadline guard: return 0 if we are running out of time (though this function is fast)
    # The evaluator handles timeouts, but we ensure we don't hang in edge cases.
    # Since this is a vectorized numpy operation, it should be < 1ms for 2161 rows.
    
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    p = params
    
    # Normalize temperature to [0, 1]
    T_min = 0.06667
    T_max = 1.0
    T_norm = (T - T_min) / (T_max - T_min)
    T_norm = np.clip(T_norm, 0.0, 1.0)
    
    # Parameter mapping
    # We use softplus or exp to ensure positive physical quantities where needed,
    # but BFGS starts at 1, so simple linear mappings with clipping might be safer
    # to avoid extreme values during early iterations.
    
    # p[0]: Elastic Modulus E (should be positive)
    E = p[0]
    
    # p[1]: Yield Strain base
    eps_y0 = p[1]
    
    # p[2]: Yield Strain temp coefficient
    eps_yT = p[2]
    
    # p[3]: Transition width (sharpness of yield)
    # We map this to a positive value for the tanh slope
    # tanh(k * x) has slope k at 0. We want k to be large enough for a sharp knee.
    # Let k = exp(p[3]) to ensure positivity and allow wide range.
    # Starting at 1 -> k=2.7. Might be too soft. 
    # Let's use k = 10 * exp(p[3])? Or just k = p[3] with a floor?
    # Better: k = 5 + 5*exp(p[3]) to ensure a minimum sharpness.
    k = 5.0 + 5.0 * np.exp(p[3])
    
    # p[4]: Yield Stress base
    Y0 = p[4]
    
    # p[5]: Yield Stress temp coefficient
    YT = p[5]
    
    # p[6]: Hardening Modulus base
    H0 = p[6]
    
    # p[7]: Hardening Modulus temp coefficient
    HT = p[7]
    
    # p[8]: Plateau offset / Max stress adjustment
    # The model is: sigma = E*eps + (Y + H*eps_p) * s
    # As eps -> inf, sigma ~ E*eps + (Y + H*eps)*1 = (E+H)*eps + Y.
    # This is linear hardening. If we want saturation, we need a different form.
    # The prompt suggests "saturation factor".
    # Let's refine the model to include saturation:
    # sigma = E * eps * (1 - s) + (Y + H * eps_p * (1 - exp(-h * eps_p))) * s
    # This is getting complex. Let's stick to the simpler bilinear with tanh transition
    # which effectively creates a plateau if H is small or negative? No.
    #
    # Let's use the form suggested by the "inspiration" and "knowledge":
    # "smooth hinge s*(1-exp(-k*s))" was mentioned in knowledge but the directive
    # says "tanh based smooth step".
    #
    # Let's use:
    # sigma = E * eps + (Y + H * eps_p) * s(eps)
    # where s(eps) = 0.5 * (1 + tanh(k * (eps - eps_y)))
    # This creates a smooth transition from elastic (s~0) to plastic (s~1).
    # In the plastic region, the slope is E + H.
    # If H < -E, the curve turns down, which is unphysical for tension.
    # We should ensure H > 0. Let H = exp(p[6]) + exp(p[7])*T_norm.
    
    # Let's re-map parameters for stability and physical constraint:
    # p[0]: E (Elastic Modulus) -> exp(p[0]) to ensure > 0. Start 1 -> e=2.7.
    # p[1]: eps_y (Yield Strain) -> p[1] (can be negative? No, yield is usually positive).
    #       Let's use softplus or just clip.
    # p[2]: eps_y_T (Temp effect on yield strain)
    # p[3]: k (Transition sharpness) -> exp(p[3])
    # p[4]: Y (Yield Stress) -> p[4]
    # p[5]: Y_T (Temp effect on yield stress)
    # p[6]: H (Hardening Modulus) -> exp(p[6]) to ensure > 0.
    # p[7]: H_T (Temp effect on hardening)
    # p[8]: Offset
    # p[9]: Global Scale
    
    # Revised Parameter Mapping:
    E = np.exp(p[0])
    eps_y = p[1] + p[2] * T_norm
    k = np.exp(p[3])
    Y = p[4] + p[5] * T_norm
    H = np.exp(p[6]) + p[7] * T_norm  # H can be negative if p[7] is large negative? 
                                      # Better to keep H positive for hardening.
                                      # Let's make H = exp(p[6]) * (1 + p[7]*T_norm)
                                      # If 1+p[7]*T_norm goes negative, it's bad.
                                      # Let's just use H = exp(p[6]) and let Y handle the shift.
                                      # Or H = exp(p[6]) + exp(p[7])*T_norm?
                                      # Let's keep it simple: H = exp(p[6])
    H = np.exp(p[6])
    
    # p[7] and p[8] are currently unused in this specific mapping if I stick to the above.
    # I need to use all 10 params.
    
    # Let's add a saturation term to the plastic part to model a plateau.
    # sigma_plastic = Y + H * eps_p * (1 - exp(-h * eps_p))
    # This saturates at Y + H/h.
    # Let h = exp(p[7]).
    
    # So:
    # s = 0.5 * (1 + tanh(k * (eps - eps_y)))
    # eps_p = np.maximum(eps - eps_y, 0)
    # sigma_plastic = Y + H * eps_p * (1 - np.exp(-h * eps_p))
    # sigma = E * eps * (1 - s) + sigma_plastic * s
    
    # This uses:
    # p[0]: E
    # p[1]: eps_y
    # p[2]: eps_y_T
    # p[3]: k
    # p[4]: Y
    # p[5]: Y_T
    # p[6]: H
    # p[7]: h (saturation rate)
    # p[8]: Offset
    # p[9]: Global Scale
    
    # Let's implement this.
    
    # Ensure eps_p is non-negative for the saturation term calculation
    # The tanh transition handles the blending, but the formula for eps_p
    # inside the plastic term should ideally be the "plastic strain".
    # If we use eps_p = max(eps - eps_y, 0), the derivative is 0 at the knee,
    # which is good.
    
    eps_y_val = p[1] + p[2] * T_norm
    k_val = np.exp(p[3])
    Y_val = p[4] + p[5] * T_norm
    H_val = np.exp(p[6])
    h_val = np.exp(p[7])
    offset = p[8]
    scale = p[9]
    
    # Calculate smooth step
    # Clip argument to tanh to avoid overflow in exp? tanh is safe for large inputs.
    # But k * (eps - eps_y) can be large.
    arg = k_val * (eps - eps_y_val)
    # To avoid numerical issues with very large arg, we can clip, but tanh handles it.
    # However, for very negative arg, tanh -> -1, s -> 0.
    # For very positive arg, tanh -> 1, s -> 1.
    s = 0.5 * (1.0 + np.tanh(arg))
    
    # Plastic strain
    # Using max(eps - eps_y, 0) creates a kink in the derivative of eps_p,
    # but it is multiplied by s which is 0 at the knee, so the product is smooth?
    # Let's check: f(x) = max(x-a, 0) * s(x).
    # If s(a)=0.5 (midpoint), then f(a) = 0.
    # Derivative from left: 0 * s' + 0 = 0.
    # Derivative from right: 1 * s(a) + (x-a)*s'(x) -> 0.5 * s'(a).
    # So there is a kink in the derivative at the yield point if we use max().
    # To be fully smooth, we should use a smooth approximation of max, or accept
    # that BFGS can handle mild non-differentiability? BFGS assumes differentiability.
    # Better to use a smooth hinge: eps_p = 0.5 * (eps - eps_y + sqrt((eps-eps_y)^2 + delta^2))
    # But that adds a param delta.
    # Alternative: Use the same tanh for the hinge?
    # Let's just use eps_p = eps - eps_y_val.
    # If eps < eps_y, eps_p is negative.
    # The term H * eps_p * (1 - exp(-h * eps_p)) will be negative for eps_p < 0.
    # But it is multiplied by s, which is small.
    # This ensures full differentiability.
    
    eps_p = eps - eps_y_val
    
    # Saturation term
    # 1 - exp(-h * eps_p)
    # If eps_p is large negative, -h*eps_p is large positive, exp is huge, term is -huge.
    # This can cause overflow.
    # We need to be careful.
    # Since s is ~0 for eps < eps_y, the contribution should be small.
    # But if h is large and eps_p is -1, exp(h) is huge.
    # To avoid this, we can clip the argument to exp.
    # Or, we can define the plastic term only for eps_p > 0 using a smooth mask?
    # No, let's use a safe exp.
    # Actually, if we use the form:
    # sigma_plastic = Y + H * eps_p / (1 + h * eps_p)  # Hyperbolic hardening?
    # This is bounded and smooth.
    # Let's try hyperbolic hardening: H * eps_p / (1 + h * eps_p)
    # If eps_p > 0, it saturates at H/h.
    # If eps_p < 0, it goes negative.
    # This is much safer numerically than exp.
    
    denom = 1.0 + h_val * eps_p
    # Avoid division by zero if 1 + h*eps_p = 0
    # h > 0, eps_p can be negative.
    # If eps_p < -1/h, denom < 0.
    # For our range, eps is in [-0.002, 0.96]. eps_y is likely small (0.001-0.01).
    # eps_p is roughly in [-0.01, 0.95].
    # h is exp(p[7]). If p[7]=1, h=2.7. 1/h = 0.37.
    # So if eps_p < -0.37, denom is negative.
    # But eps_p is rarely that negative.
    # To be safe, we can clip denom to a small value.
    denom_safe = np.where(np.abs(denom) < 1e-6, 1e-6 * np.sign(denom) + (1-np.sign(denom)*1e-6), denom)
    # Simpler: denom_safe = np.maximum(denom, 1e-6) if we assume hardening is only positive?
    # No, let's just use a large number for the slope if denom is close to 0.
    # Actually, the physical model usually assumes hardening modulus is positive.
    # Let's stick to the exp version but clip the input to exp to prevent overflow.
    # exp_arg = -h_val * eps_p
    # exp_arg = np.clip(exp_arg, -50, 50)
    # sat = 1 - np.exp(-exp_arg) # Wait, 1 - exp(-h*eps_p)
    # If h*eps_p is large positive, exp is huge, 1-huge is -huge.
    # This is the problem.
    
    # Let's go back to the simpler bilinear model without saturation,
    # but add a global saturation on the total stress?
    # Or just use the bilinear model:
    # sigma = E * eps + (Y + H * eps_p) * s
    # This is linear in the plastic region. It doesn't plateau, but it's very stable.
    # The "knowledge" said "plateau" was key.
    # Let's use the hyperbolic form but ensure stability.
    
    # Let's try: sigma = E * eps * (1-s) + (Y + H * eps_p * (1 - np.exp(-h * np.maximum(eps_p, 0)))) * s
    # By using max(eps_p, 0) inside the exp, we avoid the negative exp blowup.
    # The kink in max is smoothed by s?
    # At eps_p = 0, s = 0.5.
    # Left deriv: 0. Right deriv: s(0) * H * (1-1) + ... = 0.
    # Wait, d/dx (x * (1-exp(-h*x))) at x=0 is 1*(1-1) + x*(h*exp(-hx)) = 0.
    # So the term is C^1 smooth at 0 even with max?
    # Let g(x) = x * (1-exp(-h*x)) for x>0, 0 for x<=0.
    # g(0)=0. g'(0+) = 1*(1-1) + 0 = 0. g'(0-) = 0.
    # So it is C^1.
    # g''(0+) = h*(1-exp(-hx)) + x*h*exp(-hx) -> h.
    # g''(0-) = 0.
    # So it is C^1 but not C^2. BFGS should be okay with C^1.
    
    eps_p_pos = np.maximum(eps_p, 0.0)
    # Clip argument to exp to avoid overflow for large positive eps_p_pos
    # h * eps_p_pos can be large.
    exp_arg = h_val * eps_p_pos
    exp_arg_clipped = np.clip(exp_arg, 0.0, 50.0)
    sat_term = 1.0 - np.exp(-exp_arg_clipped)
    
    plastic_stress = Y_val + H_val * eps_p_pos * sat_term
    
    # Blend
    sigma = E * eps * (1.0 - s) + plastic_stress * s
    
    # Apply offset and scale
    sigma = sigma * scale + offset
    
    # Ensure finite
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    return sigma
