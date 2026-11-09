"""E. coli growth rate model using a sum-of-sines (Fourier) basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a Fourier (sum-of-sines) basis.
    
    This model uses sinusoidal features with learnable frequencies and phases
    to capture periodic/oscillatory constraints in temperature and pH, which
    are characteristic of biological growth curves.
    
    Inputs:
    b: Population density (range ~[1, 5])
    s: Substrate concentration (range ~[1, 5])
    temp: Temperature (range ~[1, 40])
    pH: pH level (range ~[2, 12])
    
    Parameters (10):
    p0: Bias
    p1: Amplitude for b sine term
    p2: Amplitude for s sine term
    p3: Amplitude for temp sine term
    p4: Amplitude for pH sine term
    p5: Frequency for temp sine
    p6: Phase for temp sine
    p7: Frequency for pH sine
    p8: Phase for pH sine
    p9: Amplitude for b*s interaction sine term
    """
    # Unpack parameters
    bias = params[0]
    amp_b = params[1]
    amp_s = params[2]
    amp_temp = params[3]
    amp_ph = params[4]
    freq_temp = params[5]
    phase_temp = params[6]
    freq_ph = params[7]
    phase_ph = params[8]
    amp_interaction = params[9]
    
    # Normalize inputs to reasonable ranges for the sine functions
    # b: [1, 5] -> normalize to [0, 1]
    b_norm = (b - 1.0) / 4.0
    # s: [1, 5] -> normalize to [0, 1]
    s_norm = (s - 1.0) / 4.0
    # temp: [1, 40] -> normalize to [0, 1]
    temp_norm = (temp - 1.0) / 39.0
    # pH: [2, 12] -> normalize to [0, 1]
    ph_norm = (pH - 2.0) / 10.0
    
    # Construct sine features with learnable frequencies and phases
    # For temperature: sin(freq_temp * temp_norm * 2*pi + phase_temp)
    temp_sine = np.sin(freq_temp * temp_norm * 2.0 * np.pi + phase_temp)
    
    # For pH: sin(freq_ph * ph_norm * 2*pi + phase_ph)
    ph_sine = np.sin(freq_ph * ph_norm * 2.0 * np.pi + phase_ph)
    
    # For b and s: use fixed frequency (1) since we only have amplitude params
    b_sine = np.sin(b_norm * 2.0 * np.pi)
    s_sine = np.sin(s_norm * 2.0 * np.pi)
    
    # Interaction term: product of temp and pH sines
    interaction_sine = temp_sine * ph_sine
    
    # Combine all features
    pred = (bias 
            + amp_b * b_sine 
            + amp_s * s_sine 
            + amp_temp * temp_sine 
            + amp_ph * ph_sine 
            + amp_interaction * interaction_sine)
    
    # Ensure the output is finite to prevent crashing the evaluator
    return np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
