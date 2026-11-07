"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a Quadratic Polynomial model.
    
    This model uses a linear combination of polynomial features up to degree 2,
    specifically chosen to capture curvature in temperature and pH, and key
    interactions with substrate, while remaining convex-friendly for BFGS.
    
    Parameters:
    p0: Bias
    p1: Coeff for b
    p2: Coeff for s
    p3: Coeff for temp
    p4: Coeff for pH
    p5: Coeff for temp^2
    p6: Coeff for pH^2
    p7: Coeff for temp * pH
    p8: Coeff for s * temp
    p9: Coeff for s * pH
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Compute features
    # Note: Inputs are assumed to be in the range provided. 
    # Normalization is not applied inside the equation to keep it simple and fast,
    # relying on BFGS to scale parameters appropriately if needed, 
    # though typically inputs should be normalized. 
    # Given the "convex quadratic" insight, we assume the evaluator handles scaling 
    # or the ranges are small enough.
    
    # If inputs are large, scaling helps BFGS. 
    # Let's add simple scaling based on known ranges to improve conditioning.
    # b: [1, 5] -> center 3, scale 2
    # s: [1, 40] -> center 20, scale 15
    # temp: [1, 12] -> center 6.5, scale 5
    # pH: [1, 12] -> center 6.5, scale 5
    
    # However, the prompt says "Inputs in training range...". 
    # If we scale, the fitted params will be different, but the fit should be better conditioned.
    # Let's use standardized inputs to help BFGS.
    
    b_norm = (b - 3.0) / 2.0
    s_norm = (s - 20.0) / 15.0
    temp_norm = (temp - 6.5) / 5.0
    ph_norm = (pH - 6.5) / 5.0

    # Polynomial terms
    term_1 = b_norm
    term_2 = s_norm
    term_3 = temp_norm
    term_4 = ph_norm
    term_5 = temp_norm * temp_norm
    term_6 = ph_norm * ph_norm
    term_7 = temp_norm * ph_norm
    term_8 = s_norm * temp_norm
    term_9 = s_norm * ph_norm

    # Combine
    mu = p0 + p1 * term_1 + p2 * term_2 + p3 * term_3 + p4 * term_4 + \
         p5 * term_5 + p6 * term_6 + p7 * term_7 + p8 * term_8 + p9 * term_9

    return mu
# EVOLVE-BLOCK-END
