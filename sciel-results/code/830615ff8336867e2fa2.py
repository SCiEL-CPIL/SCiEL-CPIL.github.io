"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Trigonometric basis approach
    # This uses sine and cosine functions which are well-suited for bounded inputs
    # and can capture periodic or complex non-polynomial behaviors.
    
    p = params
    
    # Scale inputs to [0, 1] or keep in original range?
    # Original range [0.5, 2]. Period 1 is reasonable.
    # Let's use 2*pi * input to have one full period across the range roughly.
    # Actually, 2*pi * x where x in [0.5, 2] gives phase [pi, 4pi].
    
    term_0 = np.ones_like(x)
    term_1 = np.sin(2 * np.pi * x)
    term_2 = np.cos(2 * np.pi * x)
    term_3 = np.sin(2 * np.pi * y)
    term_4 = np.cos(2 * np.pi * y)
    term_5 = np.sin(2 * np.pi * z)
    term_6 = np.cos(2 * np.pi * z)
    
    # Interaction terms
    term_7 = np.sin(2 * np.pi * (x + y))
    term_8 = np.cos(2 * np.pi * (x + y))
    term_9 = np.sin(2 * np.pi * (y + z))
    
    return (
        p[0] * term_0
        + p[1] * term_1
        + p[2] * term_2
        + p[3] * term_3
        + p[4] * term_4
        + p[5] * term_5
        + p[6] * term_6
        + p[7] * term_7
        + p[8] * term_8
        + p[9] * term_9
    )
# EVOLVE-BLOCK-END
