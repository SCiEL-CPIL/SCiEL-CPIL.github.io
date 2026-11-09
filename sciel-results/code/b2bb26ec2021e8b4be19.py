"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a Fourier series basis in strain with temperature-dependent coefficients.
    """
    start_time = time.time()
    deadline = 145.0 # 5s margin from 150s limit
    
    # Normalize strain to [0, 1] range if it's not already, but inputs are in training range.
    # Training range: [-0.001911, 0.9636].
    # We will map this to [0, pi] for the Fourier terms.
    # min_s = -0.001911, max_s = 0.9636
    # s_norm = (strain - min_s) / (max_s - min_s)
    # theta = np.pi * s_norm
    
    # However, using fixed normalization constants in the equation is risky if the 
    # evaluator passes data slightly outside or if the "training range" provided 
    # is just an estimate. But the task says "Inputs in training range...".
    # To be safe and robust, I can use the raw strain directly in a scaled way.
    # Let's use theta = np.pi * strain. Since strain is ~ [0, 1], theta is ~ [0, pi].
    
    # Params:
    # p0: offset
    # p1: temp offset coeff
    # p2, p3: cos(pi*eps) coeff (const, temp)
    # p4, p5: sin(pi*eps) coeff (const, temp)
    # p6, p7: cos(2*pi*eps) coeff (const, temp)
    # p8, p9: sin(2*pi*eps) coeff (const, temp)
    
    # This uses 10 params.
    
    eps = strain
    T = temp
    
    # Pre-compute basis functions
    # Using np.pi * eps
    theta = np.pi * eps
    
    cos_1 = np.cos(theta)
    sin_1 = np.sin(theta)
    cos_2 = np.cos(2 * theta)
    sin_2 = np.sin(2 * theta)
    
    # Build the expression
    # term_0 = p0 + p1 * T
    # term_1 = (p2 + p3 * T) * cos_1
    # term_2 = (p4 + p5 * T) * sin_1
    # term_3 = (p6 + p7 * T) * cos_2
    # term_4 = (p8 + p9 * T) * sin_2
    
    pred = (params[0] + params[1] * T) + \
           (params[2] + params[3] * T) * cos_1 + \
           (params[4] + params[5] * T) * sin_1 + \
           (params[6] + params[7] * T) * cos_2 + \
           (params[8] + params[9] * T) * sin_2
           
    # Check for non-finite values
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(pred)
        
    return pred
# EVOLVE-BLOCK-END
