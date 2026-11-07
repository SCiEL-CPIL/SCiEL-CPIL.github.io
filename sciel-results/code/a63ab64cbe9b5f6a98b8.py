"""
Strategy: Physics-informed Ramberg-Osgood model with internal parameter sanitization.
This approach uses a non-linear form in parameters, which is standard for constitutive
models. To handle BFGS starting from all ones (which might be a poor start for
exponents or moduli), we apply soft constraints (clipping) to the parameters inside
the function. This ensures the function is always finite and well-behaved, allowing
BFGS to find a reasonable local minimum without crashing or diverging.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod.
    
    Model: Ramberg-Osgood with temperature correction.
    sigma = E_eff * eps + K * (E_eff * eps / sigma_y_eff)^n * sigma_y_eff
    
    Parameters (10):
    p[0]: Base Elastic Modulus (E)
    p[1]: Base Yield Stress (sigma_y)
    p[2]: Hardening Coefficient (K)
    p[3]: Hardening Exponent (n)
    p[4]: Temperature coefficient for E (alpha_E)
    p[5]: Temperature coefficient for sigma_y (alpha_y)
    p[6]: Temperature coefficient for K (alpha_K)
    p[7]: Offset stress (bias)
    p[8]: Strain offset (eps0)
    p[9]: Scaling factor for overall stress
    
    Note: Inputs are normalized/assumed to be in the range provided by the task.
    strain: [-0.001911, 0.9636]
    temp: [0.06667, 1]
    """
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Sanitize parameters to prevent numerical issues during optimization.
    # BFGS starts at 1.0. We want to allow exploration but keep values safe.
    
    # Elastic Modulus E: Should be positive. 
    # If p[0] is 1.0, it's a small value for Al (usually ~70 GPa), but the scale
    # depends on the stress units. The target stress is ~0.9, strain ~1.0.
    # So E ~ 1.0 is reasonable if stress is normalized or in MPa/100.
    # We use softplus-like or just clipping to ensure positivity and avoid 0.
    E_base = np.clip(p[0], 1e-6, 1e6)
    
    # Yield Stress: Positive.
    sy_base = np.clip(p[1], 1e-6, 1e6)
    
    # Hardening Coeff K: Positive.
    K_base = np.clip(p[2], 1e-6, 1e6)
    
    # Hardening Exponent n: Must be > 0. Typically 5-20 for metals.
    # Starting at 1.0 is low, but BFGS can move it.
    # Clip to [0.5, 100] to avoid extreme non-linearity or flatness.
    n_exp = np.clip(p[3], 0.5, 100.0)
    
    # Temperature coefficients: Can be negative (modulus decreases with T).
    # T is in [0.06, 1].
    # alpha_E, alpha_y, alpha_K
    alpha_E = p[4]
    alpha_y = p[5]
    alpha_K = p[6]
    
    # Bias and offsets
    bias = p[7]
    eps0 = p[8]
    scale = p[9]
    
    # Effective properties at temperature T
    # Linear temperature dependence: Prop(T) = Prop_base * (1 + alpha * T)
    # To ensure positivity, we might need to be careful if alpha is very negative.
    # Given T range is small [0.06, 1], and starting alpha=1, (1+1*1)=2, (1+1*0.06)=1.06.
    # If BFGS drives alpha to -10, 1 - 10*1 = -9 -> Negative E.
    # Let's constrain the term (1 + alpha*T) to be positive.
    
    # Calculate multipliers
    mult_E = 1.0 + alpha_E * T
    mult_y = 1.0 + alpha_y * T
    mult_K = 1.0 + alpha_K * T
    
    # Ensure multipliers are not zero or negative to keep E, sy, K positive.
    # If they become negative, the physics breaks. We clip them to a small positive value.
    # This creates a non-differentiable boundary, but BFGS often handles it or bounces off.
    # Alternatively, use softplus on the multiplier.
    # Let's use a safe floor.
    mult_E = np.maximum(mult_E, 1e-6)
    mult_y = np.maximum(mult_y, 1e-6)
    mult_K = np.maximum(mult_K, 1e-6)
    
    E_eff = E_base * mult_E
    sy_eff = sy_base * mult_y
    K_eff = K_base * mult_K
    
    # Adjust strain for offset
    eps_adj = eps - eps0
    
    # Ramberg-Osgood term
    # sigma_plastic = K_eff * (E_eff * eps_adj / sy_eff)^n_exp * sy_eff
    # Note: If eps_adj is negative (compression), the power law with non-integer n
    # will produce NaNs for negative base.
    # Stress-strain for a rod usually implies tension in this context given the range.
    # However, strain starts at -0.0019.
    # We should handle negative strain. 
    # Standard Ramberg-Osgood is for magnitude.
    # Let's assume the model is for tension and we take abs for the power term,
    # then apply sign of strain? Or just assume eps_adj > 0 for the main body.
    # Given the range [-0.0019, 0.96], most data is tensile.
    # To be safe and differentiable, we can use:
    # term = (E_eff * eps_adj / sy_eff)
    # If term < 0, the plastic term is undefined for non-integer n.
    # We can use np.power(np.abs(term), n_exp) * np.sign(term)
    # Or, simpler: The elastic term handles small strains. The plastic term dominates at large strains.
    # Let's use the absolute value for the base of the power, multiplied by the sign of the strain.
    # This preserves the S-shape or monotonicity if needed, though Al is usually symmetric.
    
    base_ro = (E_eff * eps_adj) / sy_eff
    # Avoid 0^negative or similar issues. base_ro is 0 when eps_adj is 0.
    # 0^n = 0 for n>0.
    
    # Use sign-preserving power
    # np.sign(x) * np.abs(x)^n
    # If n is not integer, np.power(negative, n) is NaN.
    # So we do:
    plastic_term = K_eff * (np.abs(base_ro) ** n_exp) * np.sign(base_ro) * sy_eff
    
    # Elastic term
    elastic_term = E_eff * eps_adj
    
    # Total stress
    sigma = elastic_term + plastic_term
    
    # Apply scaling and bias
    sigma = scale * sigma + bias
    
    # Final safety check
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    return sigma
# EVOLVE-BLOCK-END
