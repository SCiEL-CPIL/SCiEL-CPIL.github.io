"""Rewritten program: piecewise-linear growth plateau with rational inhibition."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.
    
    Model components:
    1. Temperature factor: piecewise-linear ramp that is zero below T_min and above T_max,
       linearly increasing from T_min to T_opt and decreasing from T_opt to T_max.
       This enforces zero growth outside viable temperature range.
    2. pH factor: similar piecewise-linear ramp for pH, zero outside viable range.
    3. Substrate factor: Monod-like saturation s / (Ks + s), ensuring diminishing returns.
    4. Density inhibition: 1 / (1 + b/Ki), representing crowding effects.
    
    params (length 10):
    params[0]: max growth rate multiplier
    params[1]: T_min (temperature lower bound for growth)
    params[2]: T_opt (temperature optimum)
    params[3]: T_max (temperature upper bound for growth)
    params[4]: pH_min (pH lower bound)
    params[5]: pH_opt (pH optimum)
    params[6]: pH_max (pH upper bound)
    params[7]: Ks (substrate half-saturation constant)
    params[8]: Ki (density inhibition constant)
    params[9]: offset (baseline growth rate)
    """
    p = params
    
    # Ensure parameters are positive where needed to avoid numerical issues
    # Use softplus-like clamping for positive parameters
    Ks = np.abs(p[7]) + 1e-6
    Ki = np.abs(p[8]) + 1e-6
    
    # Temperature factor: piecewise linear
    # Zero below T_min, rises to 1 at T_opt, falls to 0 at T_max
    T_min = p[1]
    T_opt = p[2]
    T_max = p[3]
    
    # Ensure ordering: T_min < T_opt < T_max
    T_min_eff = np.minimum(T_min, T_opt - 1e-3)
    T_max_eff = np.maximum(T_max, T_opt + 1e-3)
    
    # Rising slope from T_min to T_opt
    slope_up = 1.0 / (T_opt - T_min_eff) if (T_opt - T_min_eff) > 1e-6 else 1.0
    # Falling slope from T_opt to T_max
    slope_down = -1.0 / (T_max_eff - T_opt) if (T_max_eff - T_opt) > 1e-6 else -1.0
    
    temp_factor = np.where(
        temp < T_min_eff,
        0.0,
        np.where(
            temp <= T_opt,
            slope_up * (temp - T_min_eff),
            np.where(
                temp <= T_max_eff,
                1.0 + slope_down * (temp - T_opt),
                0.0
            )
        )
    )
    
    # pH factor: piecewise linear
    pH_min = p[4]
    pH_opt = p[5]
    pH_max = p[6]
    
    pH_min_eff = np.minimum(pH_min, pH_opt - 1e-3)
    pH_max_eff = np.maximum(pH_max, pH_opt + 1e-3)
    
    slope_up_ph = 1.0 / (pH_opt - pH_min_eff) if (pH_opt - pH_min_eff) > 1e-6 else 1.0
    slope_down_ph = -1.0 / (pH_max_eff - pH_opt) if (pH_max_eff - pH_opt) > 1e-6 else -1.0
    
    pH_factor = np.where(
        pH < pH_min_eff,
        0.0,
        np.where(
            pH <= pH_opt,
            slope_up_ph * (pH - pH_min_eff),
            np.where(
                pH <= pH_max_eff,
                1.0 + slope_down_ph * (pH - pH_opt),
                0.0
            )
        )
    )
    
    # Substrate: Monod saturation
    substrate_factor = s / (Ks + s)
    
    # Density inhibition
    density_factor = 1.0 / (1.0 + b / Ki)
    
    # Combine
    mu = p[0] * temp_factor * pH_factor * substrate_factor * density_factor + p[9]
    
    return mu
# EVOLVE-BLOCK-END
