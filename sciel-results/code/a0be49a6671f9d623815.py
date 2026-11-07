import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Chebyshev polynomial basis.
    Inputs are scaled to [-1, 1] to leverage the properties of Chebyshev polynomials.
    """
    # Scaling factors derived from training ranges
    # x in [-0.5728, 0.5275]
    # v in [-0.1754, 0.2821]
    
    # Calculate midpoints and half-widths
    x_mid = -0.02265
    x_half = 0.55015
    v_mid = 0.05335
    v_half = 0.22875
    
    # Scale to [-1, 1]
    u = (x - x_mid) / x_half
    w = (v - v_mid) / v_half
    
    # Clip to [-1, 1] to ensure stability if inputs slightly exceed training range
    # The task says "Inputs in training range", but clipping is a safe guard for BFGS steps
    u = np.clip(u, -1.0, 1.0)
    w = np.clip(w, -1.0, 1.0)
    
    # Chebyshev polynomials of the first kind
    # T0(z) = 1
    # T1(z) = z
    # T2(z) = 2z^2 - 1
    # T3(z) = 4z^3 - 3z
    # T4(z) = 8z^4 - 8z^2 + 1
    
    T0_u = np.ones_like(u)
    T1_u = u
    T2_u = 2 * u**2 - 1
    T3_u = 4 * u**3 - 3 * u
    T4_u = 8 * u**4 - 8 * u**2 + 1
    
    T0_w = np.ones_like(w)
    T1_w = w
    T2_w = 2 * w**2 - 1
    T3_w = 4 * w**3 - 3 * w
    
    # Construct the linear combination
    # We have 10 parameters.
    # p0: T0_u (Constant)
    # p1: T1_u (Linear x)
    # p2: T2_u (Quadratic x)
    # p3: T3_u (Cubic x)
    # p4: T4_u (Quartic x)
    # p5: T1_w (Linear v)
    # p6: T2_w (Quadratic v)
    # p7: T3_w (Cubic v)
    # p8: T1_u * T1_w (Cross term x*v)
    # p9: T2_u * T1_w (Cross term x^2*v)
    
    pred = (
        params[0] * T0_u +
        params[1] * T1_u +
        params[2] * T2_u +
        params[3] * T3_u +
        params[4] * T4_u +
        params[5] * T1_w +
        params[6] * T2_w +
        params[7] * T3_w +
        params[8] * (T1_u * T1_w) +
        params[9] * (T2_u * T1_w)
    )
    
    return pred
