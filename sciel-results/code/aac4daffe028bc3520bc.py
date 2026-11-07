"""
Strategy: Physics-Constrained Ramberg-Osgood Model.
Instead of a linear-in-parameters basis expansion, we use a non-linear physics model.
The model is: sigma = E(T) * eps + K(T) * (eps - eps_y(T))^n * H(eps - eps_y(T))
where H is a smooth step function (tanh) to handle the elastic-plastic transition.
Parameters are mapped to physical quantities:
p0, p1: Base and Temp coeff for Young's Modulus E
p2, p3: Base and Temp coeff for Yield Strain eps_y
p4, p5: Base and Temp coeff for Hardening Coefficient K
p6: Hardening Exponent n
p7: Transition Width (sharpness)
p8: Offset
p9: Temp Offset
This structure is highly constrained by physics, which helps BFGS navigate the non-convex landscape by providing a strong gradient direction towards the physical solution.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a Ramberg-Osgood inspired non-linear model.
    Inputs:
        strain: array of strain values
        temp: array of temperature values
        params: array of 10 physical parameters
    Output:
        array of predicted stress values
    """
    eps = np.asarray(strain, dtype=np.float64).flatten()
    T = np.asarray(temp, dtype=np.float64).flatten()
    
    # Extract parameters
    # p0: Base E
    # p1: E_temp_coeff
    # p2: Base Yield Strain
    # p3: Yield Strain Temp Coeff
    # p4: Base K (Hardening)
    # p5: K Temp Coeff
    # p6: Hardening Exponent n
    # p7: Transition Width (k)
    # p8: Bias
    # p9: Temp Bias
    
    E_base = params[0]
    E_temp = params[1]
    ey_base = params[2]
    ey_temp = params[3]
    K_base = params[4]
    K_temp = params[5]
    n_exp = params[6]
    k_width = params[7]
    bias = params[8]
    bias_T = params[9]
    
    # Ensure positivity for physical quantities to avoid domain errors
    # Use softplus for E and K to keep them positive but allow gradients
    # E = exp(log(E_base + E_temp*T)) is too restrictive? 
    # Let's just use max(param, small_positive) or exp?
    # BFGS from ones: E_base=1, E_temp=1. E(0.5) = 1.5.
    # If we use exp, E = exp(p0 + p1*T). This is always positive.
    # Let's use exp for E and K to ensure positivity.
    # For Yield Strain, it should be small positive.
    # For n, it's typically 2-8 for Al.
    
    # Model:
    # E_eff = exp(p0 + p1 * T)
    # ey_eff = exp(p2 + p3 * T)  # Ensure positive yield strain
    # K_eff = exp(p4 + p5 * T)   # Ensure positive hardening coeff
    # n_eff = p6                 # Exponent
    
    # If we use exp, the starting point (all ones) gives:
    # E(0.5) = exp(1.5) ~ 4.5
    # ey(0.5) = exp(1.5) ~ 4.5 (Too high, strain is < 1)
    # This might cause BFGS to struggle if the scale is way off.
    # The evaluator starts from all ones.
    # If we use exp, we need to scale the inputs or use a different mapping.
    # Alternative: Use softplus(x) = log(1+exp(x)).
    # softplus(1) ~ 1.31. Still a bit high for yield strain if expected ~0.02.
    
    # Let's try a linear mapping with a floor, or just rely on BFGS to find the right scale.
    # The "Linear-in-parameters" strategy worked well because it was convex.
    # This non-linear strategy is a gamble on the landscape.
    # To make it safer, let's use a form that is close to linear but allows curvature.
    
    # Revised Physics Model:
    # sigma = E_eff * eps * (1 - S) + (sigma_y + K_eff * (eps - ey)^n) * S
    # Where S is a smooth step.
    # Let's stick to the exponential mapping for E and K, but initialize parameters such that
    # if params are all 1, the values are reasonable? We can't change the start point.
    # Start point: all params = 1.
    # If we use exp(p0 + p1*T):
    # T in [0.06, 1.0].
    # E = exp(1 + 1*T) ~ exp(1.06) to exp(2). ~ 2.9 to 7.4.
    # This is a reasonable scale for stress if strain is ~0.1?
    # Wait, strain is up to 0.96. Stress up to 0.9.
    # If E ~ 5, and eps ~ 0.1, sigma ~ 0.5. Reasonable.
    # Yield strain: exp(1 + 1*T) ~ 2.9 to 7.4. This is WAY too high. Strain is < 1.
    # So exp mapping for yield strain is bad.
    
    # Better mapping for yield strain:
    # ey = 1 / (1 + exp(-(p2 + p3*T)))  # Logistic?
    # If p2=1, p3=1, T=0.5: 1/(1+exp(-1.5)) ~ 0.82. Still high.
    # We want ey ~ 0.02.
    # Maybe ey = exp(-p2 - p3*T)?
    # If p2=1, p3=1, T=0.5: exp(-1.5) ~ 0.22. Closer.
    # If p2=3, p3=0: exp(-3) ~ 0.05.
    # So using negative exponential for yield strain makes sense.
    
    # Let's define:
    # E_eff = exp(p0 + p1*T)  # Positive, scale ~ e^1 to e^2
    # ey_eff = exp(-(p2 + p3*T)) # Small positive, scale ~ e^-1 to e^-3
    # K_eff = exp(p4 + p5*T)  # Positive
    # n_eff = p6
    # S = 0.5 * (1 + tanh(k_width * (eps - ey_eff)))
    # k_width = exp(p7) # Positive sharpness
    
    # If params are all 1:
    # E ~ 3-7
    # ey ~ 0.2-0.3 (Still a bit high, but BFGS can lower p2)
    # K ~ 3-7
    # n = 1
    # k_width = e ~ 2.7 (Very smooth transition)
    
    # This seems like a reasonable starting guess for the optimizer to refine.
    
    E_eff = np.exp(params[0] + params[1] * T)
    ey_eff = np.exp(-(params[2] + params[3] * T))
    K_eff = np.exp(params[4] + params[5] * T)
    n_eff = params[6]
    k_width = np.exp(params[7])
    
    # Smooth step function S(eps, T)
    # S goes from 0 (elastic) to 1 (plastic)
    # Argument: k_width * (eps - ey_eff)
    # If eps < ey, arg is negative, tanh ~ -1, S ~ 0.
    # If eps > ey, arg is positive, tanh ~ 1, S ~ 1.
    
    # Clip the argument to avoid overflow in tanh? Tanh is safe, but exp in E/K might overflow if params get large.
    # BFGS usually keeps params reasonable.
    
    arg = k_width * (eps - ey_eff)
    # Clip arg to [-500, 500] to be safe for tanh (though tanh is fine, it's just to be safe)
    arg = np.clip(arg, -500, 500)
    S = 0.5 * (1.0 + np.tanh(arg))
    
    # Elastic term
    sigma_elastic = E_eff * eps
    
    # Plastic term: Yield Stress + Hardening
    # sigma_plastic = E_eff * ey_eff + K_eff * (eps - ey_eff)^n_eff
    # Note: (eps - ey_eff) can be negative if eps < ey.
    # But S is ~0 there, so it shouldn't matter much.
    # However, (negative)^n is undefined for non-integer n.
    # We must ensure the base is positive.
    # Use np.maximum(eps - ey_eff, 1e-10)
    
    eps_plastic = np.maximum(eps - ey_eff, 1e-10)
    
    # If n_eff is not integer, and eps_plastic is 0, it's 0.
    # If n_eff < 0, it blows up. But n is typically > 1.
    # Let's assume n_eff is reasonable.
    
    sigma_yield = E_eff * ey_eff
    sigma_hardening = K_eff * np.power(eps_plastic, n_eff)
    sigma_plastic = sigma_yield + sigma_hardening
    
    # Blend
    sigma = sigma_elastic * (1.0 - S) + sigma_plastic * S
    
    # Add biases
    sigma = sigma + params[8] + params[9] * T
    
    # Safety check
    if not np.all(np.isfinite(sigma)):
        sigma = np.where(np.isfinite(sigma), sigma, 0.0)
        
    return sigma
# EVOLVE-BLOCK-END
