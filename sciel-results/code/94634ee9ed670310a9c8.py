"""Rewritten program for E. coli growth rate symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.
    Structure: Multiplicative biological core (Monod, Gaussian T/pH, Density Inhibition) 
               + Additive linear correction term.
    
    Parameters:
    p0: log(K_s) for Monod substrate half-saturation
    p1: shift for T_opt (center 37)
    p2: log(sigma_t) for temperature width
    p3: shift for pH_opt (center 7)
    p4: log(sigma_p) for pH width
    p5: log(K_b) for density inhibition half-saturation
    p6: log(V_max) for maximum growth rate
    p7: linear coeff for b (centered at 3)
    p8: linear coeff for s (centered at 3)
    p9: linear coeff for temp (centered at 20)
    """
    # Deadline guard (minimal overhead)
    # This is a vectorized operation, so it's very fast. 
    # We just ensure we don't hang if numpy is misbehaving (unlikely).
    start_time = time.time()
    
    # Extract and transform parameters
    # Add small epsilon to prevent division by zero or log of zero if params go negative
    eps = 1e-6
    
    K_s = np.exp(params[0]) + eps
    T_opt = 37.0 + params[1]
    sigma_t = np.exp(params[2]) + eps
    pH_opt = 7.0 + params[3]
    sigma_p = np.exp(params[4]) + eps
    K_b = np.exp(params[5]) + eps
    V_max = np.exp(params[6]) + eps
    
    # Core Biological Terms
    # 1. Substrate: Monod Kinetics
    # s range [1, 5], K_s starts at ~2.7
    monod = s / (K_s + s)
    
    # 2. Temperature: Gaussian Response
    # temp range [1, 40], T_opt starts at 38
    # Clip exponent to prevent underflow warnings, though exp handles it
    z_t = (temp - T_opt) / sigma_t
    gauss_t = np.exp(-0.5 * z_t * z_t)
    
    # 3. pH: Gaussian Response
    # pH range [2, 12], pH_opt starts at 8
    z_p = (pH - pH_opt) / sigma_p
    gauss_p = np.exp(-0.5 * z_p * z_p)
    
    # 4. Population Density: Inhibition
    # b range [1, 5], K_b starts at ~2.7
    # Using Michaelis-Menten style inhibition: K_b / (K_b + b)
    inhib_b = K_b / (K_b + b)
    
    # Combined Core
    core = V_max * monod * gauss_t * gauss_p * inhib_b
    
    # Additive Correction
    # Centered to help BFGS convergence
    offset = (params[7] * (b - 3.0) + 
              params[8] * (s - 3.0) + 
              params[9] * (temp - 20.0))
    
    result = core + offset
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    # Check deadline (though this should be instant)
    if time.time() - start_time > 115:
         # Fallback to simple linear if something went wrong
         return params[0] * b + params[1] * s + params[2] * temp + params[3] * pH + params[4]
         
    return result
# EVOLVE-BLOCK-END
