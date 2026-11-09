"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [-1, 1]
    # x, y, z are in [0.5, 2.0]
    # Center = 1.25, Half-range = 0.75
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Quadratic basis terms
    # 1: constant
    # 2-4: linear
    # 5-7: squared
    # 8-10: cross terms
    term_1 = np.ones_like(u)
    term_2 = u
    term_3 = v
    term_4 = w
    term_5 = u * u
    term_6 = v * v
    term_7 = w * w
    term_8 = u * v
    term_9 = u * w
    term_10 = v * w
    
    p = params
    
    # Ensure params are finite and valid, though the evaluator handles this
    # We just compute the linear combination
    return (p[0] * term_1 + 
            p[1] * term_2 + 
            p[2] * term_3 + 
            p[3] * term_4 + 
            p[4] * term_5 + 
            p[5] * term_6 + 
            p[6] * term_7 + 
            p[7] * term_8 + 
            p[8] * term_9 + 
            p[9] * term_10)
# EVOLVE-BLOCK-END
