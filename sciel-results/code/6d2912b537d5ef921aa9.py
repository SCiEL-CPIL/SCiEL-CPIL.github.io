"""
Uses a mixed polynomial-trigonometric basis for symbolic regression.
This model is linear in the 10 parameters, allowing the external BFGS optimizer
to efficiently find the global minimum.
The basis includes constant, linear, trigonometric, interaction, and quadratic
terms to capture diverse functional forms (periodic, geometric, interactive).
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Unpack parameters for clarity
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis terms - mixed polynomial and trigonometric
    # Constant
    t00 = np.ones_like(x)
    # Linear
    t10 = x
    t01 = y
    # Trigonometric (period 2 matches domain [-1,1])
    t11 = np.sin(np.pi * x)
    t02 = np.sin(np.pi * y)
    # Interactions
    t12 = x * y
    t13 = np.sin(np.pi * x) * np.sin(np.pi * y)
    # Quadratic polynomial
    t20 = x * x
    t03 = y * y
    # Mixed quadratic
    t21 = x * x * y
    t04 = x * y * y
    
    # Linear combination
    return (p0 * t00 + 
            p1 * t10 + 
            p2 * t01 + 
            p3 * t11 + 
            p4 * t02 + 
            p5 * t12 + 
            p6 * t13 + 
            p7 * t20 + 
            p8 * t03 + 
            p9 * t21)
# EVOLVE-BLOCK-END
