"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Chebyshev polynomial basis
    in scaled position and velocity.
    
    The basis functions are chosen to capture:
    - Nonlinear restoring forces (x, x^3)
    - Nonlinear damping forces (v, v^3)
    - Interaction terms
    """
    # Scaling parameters based on training ranges
    # x in [-0.5728, 0.5275]
    # v in [-0.1754, 0.2821]
    
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    x_mid = (x_min + x_max) / 2.0
    x_half = (x_max - x_min) / 2.0
    
    v_mid = (v_min + v_max) / 2.0
    v_half = (v_max - v_min) / 2.0
    
    # Scale to [-1, 1]
    # Add small epsilon to avoid division by zero if range is 0 (not the case here)
    x_s = (x - x_mid) / x_half
    v_s = (v - v_mid) / v_half
    
    # Clip to [-1, 1] to ensure stability if test data is slightly OOD, 
    # though task says in-distribution. Chebyshev grows outside, but we expect in-range.
    # Actually, for fitting, we want the exact values. Clipping might bias the fit if OOD.
    # But the task says "Inputs in training range". We will assume in-range.
    
    # Compute Chebyshev polynomials up to order 3
    T0_x = np.ones_like(x_s)
    T1_x = x_s
    T2_x = 2 * x_s**2 - 1
    T3_x = 4 * x_s**3 - 3 * x_s
    
    T0_v = np.ones_like(v_s)
    T1_v = v_s
    T2_v = 2 * v_s**2 - 1
    T3_v = 4 * v_s**3 - 3 * v_s
    
    # Construct the 10 basis functions
    # Indices 0-9 correspond to params[0]...params[9]
    
    # 0: Constant
    phi0 = T0_x * T0_v
    
    # 1: Linear X
    phi1 = T1_x * T0_v
    
    # 2: Quadratic X
    phi2 = T2_x * T0_v
    
    # 3: Cubic X
    phi3 = T3_x * T0_v
    
    # 4: Linear V
    phi4 = T0_x * T1_v
    
    # 5: Quadratic V
    phi5 = T0_x * T2_v
    
    # 6: Cubic V
    phi6 = T0_x * T3_v
    
    # 7: Interaction X*V
    phi7 = T1_x * T1_v
    
    # 8: Interaction X^2*V
    phi8 = T2_x * T1_v
    
    # 9: Interaction X*V^2
    phi9 = T1_x * T2_v
    
    # Combine with parameters
    # params is a 1D array of length 10
    p = params
    
    result = (p[0] * phi0 + p[1] * phi1 + p[2] * phi2 + p[3] * phi3 +
              p[4] * phi4 + p[5] * phi5 + p[6] * phi6 + p[7] * phi7 +
              p[8] * phi8 + p[9] * phi9)
    
    return result
# EVOLVE-BLOCK-END
