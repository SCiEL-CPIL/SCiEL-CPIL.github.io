"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Product of two polynomials of degree 4 in a scaled variable t = x/2.
    The polynomials are expressed in the Chebyshev basis T_k(t) for better conditioning.
    P(t) = sum_{i=0}^4 p_i T_i(t)
    Q(t) = sum_{j=0}^4 q_j T_j(t)
    y = P(t) * Q(t)
    params[0:5] are p0..p4
    params[5:10] are q0..q4
    """
    # Scale x from [-2, 2] to [-1, 1]
    t = x / 2.0
    
    p = params[0:5]
    q = params[5:10]
    
    # Compute Chebyshev polynomials T_0 to T_4
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    
    # Evaluate P(t) and Q(t)
    P = p[0]*T0 + p[1]*T1 + p[2]*T2 + p[3]*T3 + p[4]*T4
    Q = q[0]*T0 + q[1]*T1 + q[2]*T2 + q[3]*T3 + q[4]*T4
    
    return P * Q
# EVOLVE-BLOCK-END
