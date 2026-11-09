"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series model with 10 parameters.
    
    We use 5 sine and 5 cosine terms with fixed frequencies.
    The frequencies are chosen to cover a range of scales:
    - Low frequency: 1 (captures overall trend)
    - Medium frequencies: 2, 3 (captures moderate oscillations)
    - Higher frequencies: 4, 5 (captures finer details)
    
    Parameters:
    - params[0:5] = amplitudes for sin(1*x), sin(2*x), sin(3*x), sin(4*x), sin(5*x)
    - params[5:10] = amplitudes for cos(1*x), cos(2*x), cos(3*x), cos(4*x), cos(5*x)
    
    This is a linear model in the parameters, so BFGS will converge to the
    global minimum for the given frequency set.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Fixed frequencies
    freqs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Compute sine and cosine terms
    # Shape: (n_points, 5)
    sin_terms = np.sin(np.outer(x, freqs))  # (n, 5)
    cos_terms = np.cos(np.outer(x, freqs))  # (n, 5)
    
    # Amplitudes from params
    sin_amps = params[0:5]   # (5,)
    cos_amps = params[5:10]  # (5,)
    
    # Combine: sum over frequency index
    # sin_terms @ sin_amps gives (n,)
    result = sin_terms @ sin_amps + cos_terms @ cos_amps
    
    return result
# EVOLVE-BLOCK-END
