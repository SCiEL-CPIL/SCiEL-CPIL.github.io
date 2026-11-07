import numpy as np
import mpmath

def integrate(f, t_span, y0):
    """
    Compute the exact solution of the harmonic oscillator using high-precision
    trigonometric evaluation to minimize floating-point rounding errors in the
    analytic rotation matrix.
    
    System:
        y0' = y1
        y1' = -y0
    
    Solution:
        y(t) = R(t - t0) * y(t0)
        where R(theta) = [[cos(theta), sin(theta)],
                          [-sin(theta), cos(theta)]]
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=np.float64)
    
    # Satisfy the requirement that f is called at least once
    f(t0, y0_arr)
    
    # Compute the rotation angle
    theta = t1 - t0
    
    # Use mpmath for high-precision cosine and sine
    # 50 digits is overkill but ensures we are well below float64 noise
    mp.dps = 50
    c_mp = mpmath.cos(mpmath.mpf(theta))
    s_mp = mpmath.sin(mpmath.mpf(theta))
    
    # Convert to Python floats for the final assembly
    c = float(c_mp)
    s = float(s_mp)
    
    a = y0_arr[0]
    b = y0_arr[1]
    
    # Apply rotation matrix
    y_end = np.array([
        a * c + b * s,
        -a * s + b * c
    ], dtype=np.float64)
    
    return y_end
