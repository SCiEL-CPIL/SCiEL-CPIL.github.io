"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a Multiplicative Mechanistic Approach.
    
    This model assumes the growth rate is the product of independent physiological
    factors:
    1. Substrate availability (Monod/Hill kinetics)
    2. Biomass density inhibition (Hill inhibition)
    3. Temperature suitability (Gaussian peak)
    4. pH suitability (Gaussian peak)
    
    This approach contrasts with the additive polynomial basis by enforcing
    biological constraints (e.g., growth is zero if substrate is zero, or if
    temperature is far from optimum).
    
    Inputs are scaled to [0, 1] range to improve numerical stability and make
    parameters interpretable relative to the input range.
    
    Parameters (10 total):
    p[0]: Amplitude (Max growth rate scale)
    p[1]: Hill coefficient for substrate (n_s)
    p[2]: Half-saturation constant for substrate (K_s, scaled)
    p[3]: Half-inhibition constant for biomass (K_b, scaled)
    p[4]: Hill coefficient for biomass inhibition (n_b)
    p[5]: Temperature peak sharpness (inverse variance)
    p[6]: Temperature peak center (scaled to [0, 1])
    p[7]: pH peak sharpness (inverse variance)
    p[8]: pH peak center (scaled to [0, 1])
    p[9]: Baseline offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale inputs to [0, 1] to keep parameters in a stable range
    # Ranges from task description:
    # b: [1, 4.998] -> approx [1, 5]
    # s: [1, 4.999] -> approx [1, 5]
    # temp: [1, 39.98] -> approx [1, 40]
    # pH: [2, 12]
    
    # Avoid division by zero or negative values in power functions
    # Clip to a small epsilon to prevent 0^negative or similar issues if params go weird
    # However, standard Hill functions handle 0 well if n > 0.
    
    b_scaled = (b - 1.0) / 4.0
    s_scaled = (s - 1.0) / 4.0
    t_scaled = (temp - 1.0) / 39.0
    ph_scaled = (pH - 2.0) / 10.0
    
    # Clip to [0, 1] to ensure we stay in the intended domain for the basis functions
    # This helps BFGS not wander into unphysical regions where the model behaves badly
    b_scaled = np.clip(b_scaled, 0.0, 1.0)
    s_scaled = np.clip(s_scaled, 0.0, 1.0)
    t_scaled = np.clip(t_scaled, 0.0, 1.0)
    ph_scaled = np.clip(ph_scaled, 0.0, 1.0)
    
    # 1. Substrate Factor: Hill Function
    # f_s = s^n / (K^n + s^n)
    # If n=1, it's Monod.
    # To avoid 0/0 if s=0 and K=0 (unlikely), we add a tiny epsilon to K
    K_s = np.maximum(p2, 1e-6)
    n_s = np.maximum(p1, 0.1) # Ensure positive Hill coefficient
    
    # Handle s=0 case: 0^positive is 0.
    term_s = (s_scaled ** n_s) / (K_s ** n_s + s_scaled ** n_s)
    
    # 2. Biomass Inhibition Factor
    # f_b = 1 / (1 + (b/K)^n)
    # If b=0, f_b=1. If b is large, f_b->0.
    K_b = np.maximum(p3, 1e-6)
    n_b = np.maximum(p4, 0.1)
    
    term_b = 1.0 / (1.0 + (b_scaled / K_b) ** n_b)
    
    # 3. Temperature Factor: Gaussian
    # f_t = exp(-sharpness * (t - center)^2)
    # sharpness > 0
    sharp_t = np.maximum(p5, 0.1)
    center_t = np.clip(p6, 0.0, 1.0)
    
    term_t = np.exp(-sharp_t * (t_scaled - center_t) ** 2)
    
    # 4. pH Factor: Gaussian
    sharp_ph = np.maximum(p7, 0.1)
    center_ph = np.clip(p8, 0.0, 1.0)
    
    term_ph = np.exp(-sharp_ph * (ph_scaled - center_ph) ** 2)
    
    # Combine
    result = p0 * term_s * term_b * term_t * term_ph + p9
    
    return result
# EVOLVE-BLOCK-END
