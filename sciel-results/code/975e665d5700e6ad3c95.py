import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model for E. coli growth rate:
    - Normalized inputs for numerical stability
    - Monod saturation for substrate s
    - Gaussian-like peaked responses for temperature and pH (fitted centers/widths)
    - Linear biomass term
    - Key interaction terms
    
    Params (10 total):
    p[0]: Amplitude
    p[1]: log(Ks) for Monod
    p[2]: Optimal Temp (in normalized space)
    p[3]: log(sigma_T) for temp gaussian
    p[4]: Optimal pH (in normalized space)
    p[5]: log(sigma_pH) for pH gaussian
    p[6]: Biomass linear coeff
    p[7]: Offset
    p[8]: Temp skew (linear correction)
    p[9]: pH skew (linear correction)
    """
    # Deadline guard
    t_start = time.time()
    
    # Normalize inputs based on training ranges
    # b: [1, 4.998] -> center 3.0, scale 2.0
    # s: [1, 4.999] -> center 3.0, scale 2.0
    # temp: [1.002, 39.98] -> center 20.5, scale 10.0
    # pH: [2, 12] -> center 7.0, scale 5.0
    
    b_norm = (b - 3.0) / 2.0
    s_norm = (s - 3.0) / 2.0
    temp_norm = (temp - 20.5) / 10.0
    pH_norm = (pH - 7.0) / 5.0
    
    # Extract parameters
    A = params[0]
    log_Ks = params[1]
    T_opt = params[2]
    log_sig_T = params[3]
    pH_opt = params[4]
    log_sig_pH = params[5]
    C_b = params[6]
    Offset = params[7]
    skew_T = params[8]
    skew_pH = params[9]
    
    # Ensure stability for exp
    eps = 1e-12
    Ks = np.exp(np.clip(log_Ks, -500, 500))
    sig_T = np.exp(np.clip(log_sig_T, -500, 500))
    sig_pH = np.exp(np.clip(log_sig_pH, -500, 500))
    
    Ks_safe = Ks + eps
    sig_T_safe = sig_T + eps
    sig_pH_safe = sig_pH + eps
    
    # Monod term for substrate s (using normalized s)
    monod = s_norm / (Ks_safe + s_norm)
    
    # Temperature term: Gaussian with skew
    z_T = (temp_norm - T_opt) / sig_T_safe
    exp_T_arg = -0.5 * z_T**2 + skew_T * z_T
    exp_T_arg = np.clip(exp_T_arg, -500, 500)
    temp_term = np.exp(exp_T_arg)
    
    # pH term: Gaussian with skew
    z_pH = (pH_norm - pH_opt) / sig_pH_safe
    exp_pH_arg = -0.5 * z_pH**2 + skew_pH * z_pH
    exp_pH_arg = np.clip(exp_pH_arg, -500, 500)
    pH_term = np.exp(exp_pH_arg)
    
    # Combine: multiplicative for temp/pH (they gate growth), additive for biomass
    # The Monod term is multiplicative with the environmental gates
    pred = A * monod * temp_term * pH_term + C_b * b_norm + Offset
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > 145.0:
        return np.zeros_like(b)
    
    return pred
