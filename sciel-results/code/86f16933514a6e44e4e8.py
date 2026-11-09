"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.
    
    This model combines:
    1. A multiplicative core based on biological mechanisms:
       - Monod-like saturation for substrate: s / (Ks + s)
       - Rational bell curve for temperature: 1 / (1 + aT * (T - Topt)^2)
       - Rational bell curve for pH: 1 / (1 + aP * (pH - pHopt)^2)
       - Density-dependent inhibition: 1 / (1 + (b/Kd)^2)
    2. An additive quadratic residual correction term on centered inputs.
       This allows the model to capture any remaining linear/quadratic trends
       or interactions that the multiplicative core might miss, while the
       multiplicative core handles the primary saturation/bell shapes.
    
    The quadratic terms are centered to aid optimization convergence.
    """
    # Extract parameters
    # p[0]: mu_max (max growth rate)
    # p[1]: Ks (substrate half-saturation)
    # p[2]: Topt (optimal temperature)
    # p[3]: aT (temperature width/inverse)
    # p[4]: pHopt (optimal pH)
    # p[5]: aP (pH width/inverse)
    # p[6]: Kd (density inhibition constant)
    # p[7]: offset (baseline)
    # p[8]: weight for substrate-temperature interaction in residual
    # p[9]: weight for pH-temperature interaction in residual
    
    mu_max = params[0]
    Ks = params[1]
    Topt = params[2]
    aT = params[3]
    pHopt = params[4]
    aP = params[5]
    Kd = params[6]
    offset = params[7]
    w_st = params[8]
    w_pt = params[9]
    
    # Safe parameters to avoid division by zero or numerical instability
    # We use a small epsilon to prevent zero division.
    # Clipping to positive values ensures the "width" parameters act as expected.
    eps = 1e-6
    Ks_safe = np.clip(Ks, eps, 1e6)
    aT_safe = np.clip(aT, eps, 1e6)
    aP_safe = np.clip(aP, eps, 1e6)
    Kd_safe = np.clip(Kd, eps, 1e6)
    
    # 1. Multiplicative Core Terms
    
    # Monod saturation for substrate
    # s range [1, 40], Ks is positive.
    monod_s = s / (Ks_safe + s)
    
    # Rational bell for Temperature
    # temp range [1, 50] approx.
    dT = temp - Topt
    temp_factor = 1.0 / (1.0 + aT_safe * dT * dT)
    
    # Rational bell for pH
    # pH range [1, 14] approx.
    dP = pH - pHopt
    ph_factor = 1.0 / (1.0 + aP_safe * dP * dP)
    
    # Density inhibition
    # b range [1, 5] approx.
    density_factor = 1.0 / (1.0 + (b / Kd_safe) ** 2)
    
    # Base growth rate
    growth = mu_max * monod_s * temp_factor * ph_factor * density_factor
    
    # 2. Additive Residual Correction (Quadratic-like)
    # We add specific interaction terms to the residual to capture non-separable effects
    # that might be missed by the purely multiplicative structure.
    # Centering is applied to help BFGS converge from all-ones initialization.
    
    b_c = b - 3.0
    s_c = s - 10.0
    t_c = temp - 30.0
    p_c = pH - 7.0
    
    # Interaction terms in the residual
    # w_st * (s_c * t_c)  -> captures if substrate effect depends on temp in a non-multiplicative way
    # w_pt * (p_c * t_c)  -> captures if pH effect depends on temp in a non-multiplicative way
    
    residual = w_st * (s_c * t_c) + w_pt * (p_c * t_c)
    
    # Combine
    pred = growth + offset + residual
    
    return pred
# EVOLVE-BLOCK-END
