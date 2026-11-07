"""Rewritten program: piecewise-linear growth plateau with rational inhibition."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model with hard piecewise-linear constraints.
    
    Uses min of two clipped linear terms for temperature and pH,
    creating a tent-shaped response with hard zeros outside viable ranges.
    
    params (length 10):
    params[0]: max growth rate
    params[1]: T_min
    params[2]: T_opt  
    params[3]: T_max
    params[4]: pH_min
    params[5]: pH_opt
    params[6]: pH_max
    params[7]: Ks
    params[8]: Ki
    params[9]: offset
    """
    p = params
    
    Ks = np.abs(p[7]) + 1e-6
    Ki = np.abs(p[8]) + 1e-6
    
    # Temperature response: hard zero outside [T_min, T_max]
    T_min, T_opt, T_max = p[1], p[2], p[3]
    
    dT1 = T_opt - T_min
    dT2 = T_max - T_opt
    
    inv_dT1 = 1.0 / np.where(np.abs(dT1) > 1e-8, dT1, 1e-8)
    inv_dT2 = 1.0 / np.where(np.abs(dT2) > 1e-8, dT2, 1e-8)
    
    rise = np.clip((temp - T_min) * inv_dT1, 0.0, 1.0)
    fall = np.clip((T_max - temp) * inv_dT2, 0.0, 1.0)
    
    temp_factor = np.minimum(rise, fall)
    
    # pH response: same structure
    pH_min, pH_opt, pH_max = p[4], p[5], p[6]
    
    dpH1 = pH_opt - pH_min
    dpH2 = pH_max - pH_opt
    
    inv_dpH1 = 1.0 / np.where(np.abs(dpH1) > 1e-8, dpH1, 1e-8)
    inv_dpH2 = 1.0 / np.where(np.abs(dpH2) > 1e-8, dpH2, 1e-8)
    
    pH_rise = np.clip((pH - pH_min) * inv_dpH1, 0.0, 1.0)
    pH_fall = np.clip((pH_max - pH) * inv_dpH2, 0.0, 1.0)
    
    pH_factor = np.minimum(pH_rise, pH_fall)
    
    # Substrate: Monod
    substrate_factor = s / (Ks + s)
    
    # Density: rational inhibition
    density_factor = 1.0 / (1.0 + b / Ki)
    
    mu = p[0] * temp_factor * pH_factor * substrate_factor * density_factor + p[9]
    
    return mu
# EVOLVE-BLOCK-END
