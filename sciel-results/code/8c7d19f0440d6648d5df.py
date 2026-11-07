"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a Multiplicative Mechanistic Approach.
    
    This model assumes the growth rate is a product of saturation and response functions
    for each input, which is more biologically plausible than additive models for
    saturation kinetics.
    
    Structure:
    y = p0 * Substrate(T) * Temp(T) * pH(T) * Biomass(T)
    
    Inputs are scaled internally to keep parameters near 1 for BFGS stability.
    
    Parameters (10 total):
    p0: Overall scale
    p1: Substrate half-saturation constant (scaled)
    p2: Temp response amplitude
    p3: Temp optimal point (scaled)
    p4: pH response amplitude
    p5: pH optimal point (scaled)
    p6: Biomass response amplitude
    p7: Biomass half-saturation constant (scaled)
    p8: Temp curvature correction
    p9: pH curvature correction
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale inputs to [0, 1] or centered to improve conditioning
    # s in [1, 5] -> s/5 in [0.2, 1]
    x_s = s / 5.0
    # temp in [1, 40] -> (temp-20)/10 in [-1.9, 2.0]
    x_t = (temp - 20.0) / 10.0
    # pH in [2, 12] -> (pH-7)/5 in [-1, 1]
    x_p = (pH - 7.0) / 5.0
    # b in [1, 5] -> b/5 in [0.2, 1]
    x_b = b / 5.0
    
    # Substrate term: Monod-like saturation
    # Using x_s to keep p1 near 1
    term_s = x_s / (p1 + x_s)
    
    # Temp term: Tanh-based sigmoid centered at p3, with curvature correction
    # tanh is smooth and bounded, good for BFGS
    temp_base = np.tanh(x_t - p3)
    temp_curv = 1.0 + p8 * (x_t - p3) ** 2
    term_t = 1.0 + p2 * temp_base * temp_curv
    
    # pH term: Tanh-based sigmoid centered at p5, with curvature correction
    ph_base = np.tanh(x_p - p5)
    ph_curv = 1.0 + p9 * (x_p - p5) ** 2
    term_p = 1.0 + p4 * ph_base * ph_curv
    
    # Biomass term: Saturation-like
    term_b = 1.0 + p6 * (x_b / (p7 + x_b))
    
    # Combine
    result = p0 * term_s * term_t * term_p * term_b
    
    return result
# EVOLVE-BLOCK-END
