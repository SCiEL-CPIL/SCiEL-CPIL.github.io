"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Check if params are valid
    if not np.all(np.isfinite(params)):
        return np.zeros(len(x))
    
    # params: [amplitude, freq_x, freq_y, phase]
    # Model: amplitude * sin(freq_x * x + freq_y * y + phase)
    amplitude = params[0]
    freq_x = params[1]
    freq_y = params[2]
    phase = params[3]
    
    # Compute the sinusoidal prediction
    pred = amplitude * np.sin(freq_x * x + freq_y * y + phase)
    
    # Ensure finite
    if not np.all(np.isfinite(pred)):
        return np.zeros(len(x))
        
    return pred
# EVOLVE-BLOCK-END
