# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = params
    t = x / 3.0
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    t5 = t4 * t
    out = (p[0] + p[1] * t + p[2] * t2 + p[3] * t3 + p[4] * t4 + p[5] * t5
           + p[6] * np.tanh(x) + p[7] * np.sin(x) + p[8] * np.cos(x)
           + p[9] * np.exp(-x * x))
    return out
# EVOLVE-BLOCK-END
