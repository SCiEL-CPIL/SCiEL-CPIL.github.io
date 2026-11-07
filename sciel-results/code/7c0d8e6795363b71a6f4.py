"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Time guard: if we're close to deadline, return a simple fallback
    if time.time() > _DEADLINE:
        return np.zeros_like(x, dtype=np.float64)
    
    x = np.asarray(x, dtype=np.float64).ravel()
    
    # params length 10: Fourier series with multiple harmonics
    # p0: DC offset (constant term)
    # p1: sin(x) amplitude
    # p2: cos(x) amplitude  
    # p3: sin(2x) amplitude
    # p4: cos(2x) amplitude
    # p5: sin(3x) amplitude
    # p6: cos(3x) amplitude
    # p7: sin(4x) amplitude
    # p8: cos(4x) amplitude
    # p9: linear term (to capture any trend)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Fourier series with harmonics 1-4 plus linear trend
    y = np.full_like(x, p0, dtype=np.float64)
    y += p1 * np.sin(x)
    y += p2 * np.cos(x)
    y += p3 * np.sin(2.0 * x)
    y += p4 * np.cos(2.0 * x)
    y += p5 * np.sin(3.0 * x)
    y += p6 * np.cos(3.0 * x)
    y += p7 * np.sin(4.0 * x)
    y += p8 * np.cos(4.0 * x)
    y += p9 * x
    
    return y
# EVOLVE-BLOCK-END
