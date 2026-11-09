import numpy as np
import time

_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # x is expected to be in [0.1, 6]
    
    # Hybrid RBF model:
    # Structure:
    # p[0]: bias (global offset)
    # p[1]: slope (global linear trend)
    # p[2]: amp1, p[3]: width1_raw
    # p[4]: amp2, p[5]: width2_raw
    # p[6]: amp3, p[7]: width3_raw
    # p[8]: amp4, p[9]: width4_raw
    
    y = params[0] + params[1] * x
    
    # Fixed centers spanning the domain [0.1, 6]
    # Chosen to cover the range with good spacing
    centers = np.array([1.2, 2.4, 3.6, 4.8])
    
    for i in range(4):
        amp = params[2 * i + 2]
        width_raw = params[2 * i + 3]
        
        # Stable positive width calculation
        # Clip to prevent overflow/underflow issues in exp
        # The range [-5, 5] allows sigma from ~0.101 to ~151.6
        # This provides flexibility for both narrow and broad bumps
        w = np.exp(np.clip(width_raw, -5.0, 5.0))
        sigma = 0.1 + w
        
        dist_sq = (x - centers[i]) ** 2
        gaussian = np.exp(-dist_sq / (2.0 * sigma * sigma))
        
        y += amp * gaussian
        
    return y
