"""
Rewritten program for symbolic regression.
Uses a hybrid model: Polynomial trend + 2 Tanh-based non-linear units.
This structure allows for both global trends and local feature capture (bumps/steps).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard variables
_start_time = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a combination of polynomial terms
    and two tanh-based non-linear units.
    
    Model:
    y = p0 * x + p1 * x^2 + p2 
      + p3 * tanh(p4 * x + p5) 
      + p6 * tanh(p7 * x + p8)
      + p9 * x^3
      
    Parameters:
    p0: Linear coefficient
    p1: Quadratic coefficient
    p2: Constant offset
    p3: Amplitude of 1st tanh unit
    p4: Slope/frequency of 1st tanh unit
    p5: Offset/phase of 1st tanh unit
    p6: Amplitude of 2nd tanh unit
    p7: Slope/frequency of 2nd tanh unit
    p8: Offset/phase of 2nd tanh unit
    p9: Cubic coefficient (for extra global shape flexibility)
    """
    # Check deadline (structural requirement, minimal overhead)
    if time.time() - _start_time > _DEADLINE - _SAFETY_MARGIN:
        # In a more complex search scenario, we would return a cached result.
        # For this vectorized closed-form expression, computation is instantaneous.
        pass

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    x2 = x * x
    x3 = x2 * x

    # Polynomial part
    poly = p0 * x + p1 * x2 + p2 + p9 * x3

    # Non-linear tanh units
    # Clip arguments to tanh to avoid potential numerical issues, though tanh is stable.
    # However, if p4 or p7 are very large, p4*x can be large. tanh handles large inputs well (saturates).
    arg1 = p4 * x + p5
    arg2 = p7 * x + p8
    
    unit1 = p3 * np.tanh(arg1)
    unit2 = p6 * np.tanh(arg2)

    return poly + unit1 + unit2
# EVOLVE-BLOCK-END
