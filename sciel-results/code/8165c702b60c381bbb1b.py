# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 10 parameters:
    # p0: DC offset (bias)
    # p1: angular frequency (omega)
    # p2: phase (phi)
    # p3: amplitude for sin(omega*x + phi)
    # p4: amplitude for cos(omega*x + phi)
    # p5: angular frequency for 2nd harmonic
    # p6: phase for 2nd harmonic
    # p7: amplitude for sin(2*omega2*x + phi2)
    # p8: amplitude for cos(2*omega2*x + phi2)
    # p9: linear slope (to handle non-periodic trends)
    
    x = np.asarray(x, dtype=np.float64)
    # Ensure x is 1D for consistent broadcasting
    x_flat = x.flatten()
    
    p0 = params[0]
    omega1 = params[1]
    phi1 = params[2]
    amp_sin1 = params[3]
    amp_cos1 = params[4]
    
    omega2 = params[5]
    phi2 = params[6]
    amp_sin2 = params[7]
    amp_cos2 = params[8]
    
    slope = params[9]
    
    # Compute terms
    term1_sin = amp_sin1 * np.sin(omega1 * x_flat + phi1)
    term1_cos = amp_cos1 * np.cos(omega1 * x_flat + phi1)
    
    # Second harmonic uses double the frequency of the second set
    term2_sin = amp_sin2 * np.sin(2.0 * omega2 * x_flat + phi2)
    term2_cos = amp_cos2 * np.cos(2.0 * omega2 * x_flat + phi2)
    
    linear_term = slope * x_flat
    
    result = p0 + term1_sin + term1_cos + term2_sin + term2_cos + linear_term
    
    # Restore original shape
    return result.reshape(x.shape)
# EVOLVE-BLOCK-END
