"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sum-of-sines (Fourier) basis model for E. coli growth rate.
    
    Structural Design:
    - Uses a compact Fourier series expansion for temperature and pH,
      which are known to have optimal ranges (periodic-like constraints).
    - Uses a flexible basis for density (b) and substrate (s) that includes
      a saturation-like term (1 - exp(-s)) and linear/quadratic density terms.
    - Interactions between the periodic components and the growth-limiting
      factors are included.
    
    Parameters (length 10):
    p[0]: Bias
    p[1]: Coeff for sin(w1 * temp_n)
    p[2]: Coeff for cos(w1 * temp_n)
    p[3]: Coeff for sin(w2 * temp_n)
    p[4]: Coeff for cos(w2 * temp_n)
    p[5]: Coeff for sin(w3 * pH_n)
    p[6]: Coeff for cos(w3 * pH_n)
    p[7]: Coeff for b_n (linear density)
    p[8]: Coeff for saturation_term (1 - exp(-s_n * k))
    p[9]: Interaction coeff for temp_basis * saturation_term
    
    Note: Frequencies w1, w2, w3 are fixed to reasonable values based on
    the input ranges to ensure stability and interpretability within the
    10-parameter budget.
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Normalize inputs to [0, 1] range
    # b: [1, 5] -> [0, 1]
    b_n = (b - 1.0) / 4.0
    # s: [1, 50] -> [0, 1]
    s_n = (s - 1.0) / 49.0
    # temp: [1, 40] -> [0, 1]
    t_n = (temp - 1.0) / 39.0
    # pH: [2, 12] -> [0, 1]
    ph_n = (pH - 2.0) / 10.0
    
    # Fixed frequencies (in cycles per normalized unit)
    # Temperature optimum is often around 37C (0.9 in normalized). 
    # A low frequency captures the broad peak.
    w_t1 = 2.0 * np.pi * 1.0  # 1 cycle across the range
    w_t2 = 2.0 * np.pi * 2.0  # 2 cycles (higher freq details)
    
    # pH optimum is often around 7-8 (0.5-0.6 in normalized).
    w_ph = 2.0 * np.pi * 1.5 # 1.5 cycles across the range
    
    # Compute basis functions
    # Temperature Fourier terms
    sin_t1 = np.sin(w_t1 * t_n)
    cos_t1 = np.cos(w_t1 * t_n)
    sin_t2 = np.sin(w_t2 * t_n)
    cos_t2 = np.cos(w_t2 * t_n)
    
    # pH Fourier terms
    sin_ph = np.sin(w_ph * ph_n)
    cos_ph = np.cos(w_ph * ph_n)
    
    # Substrate saturation term (Michaelis-Menten like, but simpler)
    # Uses a fixed saturation constant to keep parameter count low
    sat_s = 1.0 - np.exp(-5.0 * s_n)
    
    # Density term
    lin_b = b_n
    
    # Interaction term: Temperature effect modulated by substrate availability
    # We use the primary temperature sine term for interaction
    inter_t_s = sin_t1 * sat_s
    
    # Construct prediction
    y = np.full_like(b, p0, dtype=float)
    
    # Add Fourier terms for Temperature
    y += p1 * sin_t1
    y += p2 * cos_t1
    y += p3 * sin_t2
    y += p4 * cos_t2
    
    # Add Fourier terms for pH
    y += p5 * sin_ph
    y += p6 * cos_ph
    
    # Add Density and Substrate terms
    y += p7 * lin_b
    y += p8 * sat_s
    
    # Add Interaction
    y += p9 * inter_t_s
    
    return y
# EVOLVE-BLOCK-END
