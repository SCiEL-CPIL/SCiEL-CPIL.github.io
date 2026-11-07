"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid rational model for E. coli growth rate.
    
    Structure:
    y = (A * S_sat + B) / (1 + C * S + D * T + E * pH)
    
    where:
    - S_sat is a Monod-like saturation term for substrate: s_norm / (s_norm + Ks)
    - S, T, pH are normalized inputs
    - The denominator provides rational saturation and interaction effects
    - The numerator allows additive combination of substrate effect and baseline
    
    This structure:
    - Captures substrate saturation via S_sat
    - Captures temperature and pH effects via rational denominator
    - Allows interactions between substrate and other factors through the ratio
    - Is smooth and differentiable everywhere (denominator kept positive)
    - Uses 10 parameters total
    """
    p = params
    
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 5] -> (b-1)/4
    # s: [1, 5] -> (s-1)/4
    # temp: [1, 40] -> (temp-1)/39
    # pH: [2, 12] -> (pH-2)/10
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    # Extract parameters
    A = p[0]          # Substrate effect amplitude
    Ks = np.exp(p[1]) # Substrate half-saturation (positive)
    B = p[2]          # Baseline in numerator
    C = p[3]          # Substrate term in denominator
    D = p[4]          # Temperature term in denominator
    E = p[5]          # pH term in denominator
    F = p[6]          # Temperature optimum offset
    G = p[7]          # pH optimum offset
    H = p[8]          # Temperature width
    I = p[9]          # pH width
    
    # Monod-like saturation for substrate
    # s_norm / (s_norm + Ks)
    # Ensure Ks is positive via exp
    S_sat = s_norm / (s_norm + Ks + 1e-8)
    
    # Temperature and pH effects via Gaussian-like terms in denominator
    # Use (x - opt)^2 / width^2 to create bell-shaped effects
    # But we want these in the denominator to create suppression at extremes
    # Actually, for a growth rate, we want suppression at non-optimal T and pH
    # So we add terms that are large at extremes, suppressing the output
    
    # Temperature suppression term
    # (t_norm - t_opt)^2 / temp_width
    # This is small at optimum, large at extremes
    # Adding to denominator suppresses y at extremes
    t_opt = 0.5 + 0.5 * np.tanh(F)  # Map F to [0, 1]
    t_width = np.exp(H) + 1e-6       # Positive width
    t_suppress = (t_norm - t_opt) ** 2 / t_width
    
    # pH suppression term
    ph_opt = 0.5 + 0.5 * np.tanh(G)  # Map G to [0, 1]
    ph_width = np.exp(I) + 1e-6      # Positive width
    ph_suppress = (ph_norm - ph_opt) ** 2 / ph_width
    
    # Numerator: A * S_sat + B
    numerator = A * S_sat + B
    
    # Denominator: 1 + C * s_norm + D * t_suppress + E * ph_suppress
    # Ensure denominator is positive
    denominator = 1.0 + C * s_norm + D * t_suppress + E * ph_suppress
    
    # Safety: ensure denominator doesn't get too close to zero
    # Use a soft floor
    denominator = np.where(np.abs(denominator) < 1e-4, 1e-4, denominator)
    
    # Final prediction
    y = numerator / denominator
    
    return y
# EVOLVE-BLOCK-END
