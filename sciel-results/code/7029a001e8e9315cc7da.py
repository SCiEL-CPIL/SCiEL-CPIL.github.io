"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a sum-of-sines (Fourier) basis.
    
    Structural change: Replace the previous mechanistic/linear basis with a
    Fourier series expansion. This allows the model to capture periodic or
    oscillatory behaviors in the growth rate response to temperature and pH,
    which are often difficult to represent with polynomials or local RBFs.
    
    The model is a linear combination of sine and cosine terms for each input,
    plus their interactions. The frequencies are fixed to cover the expected
    range of variation, and the amplitudes/phases are fitted by the optimizer.
    
    Parameters (10 used):
    p0: Bias term
    p1: Amplitude for sin(temp)
    p2: Amplitude for cos(temp)
    p3: Amplitude for sin(pH)
    p4: Amplitude for cos(pH)
    p5: Amplitude for sin(s)
    p6: Amplitude for cos(s)
    p7: Amplitude for sin(b)
    p8: Amplitude for cos(b)
    p9: Coefficient for interaction term sin(temp)*sin(pH)
    """
    # Unpack params
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalize inputs to a standard range for the Fourier terms.
    # Based on training ranges:
    # b: [1, 4.998] -> normalized to [0, 1]
    # s: [1, 39.98] -> normalized to [0, 1]
    # temp: [1, 40] -> normalized to [0, 1]
    # pH: [1, 12] -> normalized to [0, 1]
    
    b_norm = (b - 1.0) / (4.998 - 1.0)
    s_norm = (s - 1.0) / (39.98 - 1.0)
    temp_norm = (temp - 1.0) / (40.0 - 1.0)
    ph_norm = (pH - 1.0) / (12.0 - 1.0)
    
    # Define frequencies. 
    # Using a base frequency of 2*pi to capture one full cycle over the normalized range.
    # We might need higher frequencies if the response is more complex.
    # Let's start with fundamental frequency for each.
    omega = 2.0 * np.pi
    
    # Compute Fourier terms
    # For each variable, we have sin and cos components.
    
    # Temperature terms
    sin_t = np.sin(omega * temp_norm)
    cos_t = np.cos(omega * temp_norm)
    
    # pH terms
    sin_p = np.sin(omega * ph_norm)
    cos_p = np.cos(omega * ph_norm)
    
    # Substrate terms
    sin_s = np.sin(omega * s_norm)
    cos_s = np.cos(omega * s_norm)
    
    # Density terms
    sin_b = np.sin(omega * b_norm)
    cos_b = np.cos(omega * b_norm)
    
    # Interaction term: product of temp and pH sine terms
    # This captures the interaction between temperature and pH optima.
    interaction = sin_t * sin_p
    
    # Combine linearly
    result = (p0 + 
              p1 * sin_t + 
              p2 * cos_t + 
              p3 * sin_p + 
              p4 * cos_p + 
              p5 * sin_s + 
              p6 * cos_s + 
              p7 * sin_b + 
              p8 * cos_b + 
              p9 * interaction)
    
    # Ensure finite output
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
