"""E. coli growth rate model with multiplicative biological core and additive correction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a multiplicative biological core.
    
    Structure:
        Core = V_max * Monod(s) * TempFactor(T) * pHFactor(pH) * DensityFactor(b)
        Prediction = Core + LinearCorrection
    
    Parameters (10 total):
    p0: log(K_s) for Monod substrate half-saturation
    p1: Temperature center shift (added to 37.0)
    p2: log(sigma_t) for temperature width
    p3: pH center shift (added to 7.0)
    p4: log(sigma_p) for pH width
    p5: log(K_b) for density inhibition half-saturation
    p6: log(V_max) for maximum growth rate
    p7: linear coeff for b (centered at 3)
    p8: linear coeff for s (centered at 3)
    p9: linear coeff for temp (centered at 20)
    
    Returns:
        Array of predicted growth rates
    """
    # Deadline guard
    start_time = time.time()
    
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure numerical stability for exponential parameters
    eps = 1e-6
    
    # Positive scale parameters via exp
    K_s = np.exp(p0) + eps
    sigma_t = np.exp(p2) + eps
    sigma_p = np.exp(p4) + eps
    K_b = np.exp(p5) + eps
    V_max = np.exp(p6) + eps
    
    # Centers
    T_center = 37.0 + p1
    pH_center = 7.0 + p3
    
    # 1. Substrate: Monod Kinetics
    # s range [1, 5]
    monod = s / (K_s + s)
    
    # 2. Temperature: Smooth Gaussian Response
    # temp range [1, 40]
    # Clip z to prevent overflow in exp, though exp handles large negs well
    z_t = (temp - T_center) / sigma_t
    # Clip to avoid extreme underflow/overflow issues in gradient calculation
    z_t = np.clip(z_t, -10, 10)
    gauss_t = np.exp(-0.5 * z_t * z_t)
    
    # 3. pH: Smooth Gaussian Response
    # pH range [2, 12]
    z_p = (pH - pH_center) / sigma_p
    z_p = np.clip(z_p, -10, 10)
    gauss_p = np.exp(-0.5 * z_p * z_p)
    
    # 4. Population Density: Inhibition
    # b range [1, 5]
    # Michaelis-Menten style inhibition: K_b / (K_b + b)
    inhib_b = K_b / (K_b + b)
    
    # Combined Core
    core = V_max * monod * gauss_t * gauss_p * inhib_b
    
    # Additive Correction
    # Centered to help BFGS convergence and capture linear residuals
    # Using centered inputs to improve conditioning
    offset = (p7 * (b - 3.0) + 
              p8 * (s - 3.0) + 
              p9 * (temp - 20.0))
    
    result = core + offset
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    # Check deadline
    if time.time() - start_time > 115:
         # Fallback to simple linear if something went wrong (unlikely for this vectorized op)
         return params[0] * b + params[1] * s + params[2] * temp + params[3] * pH + params[4]
         
    return result
# EVOLVE-BLOCK-END
