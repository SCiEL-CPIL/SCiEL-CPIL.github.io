import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    
    Structural changes:
    1. Scaled inputs x and v to O(1) magnitude to improve condition number.
    2. Basis includes v^3 (cubic damping) instead of x^2, which is more physical for symmetric oscillators.
    3. Basis: 1, xs, xs^3, vs, vs^3, xs^2*vs, xs*vs, sin(omega*t), cos(omega*t)
       where xs = x/x_scale, vs = v/v_scale.
    4. omega is a fitted parameter (params[9]).
    
    Params:
    p0: bias
    p1: linear spring (scaled)
    p2: cubic spring (scaled)
    p3: linear damping (scaled)
    p4: cubic damping (scaled)
    p5: x^2 v damping (scaled)
    p6: x v damping (scaled)
    p7: sin drive amp
    p8: cos drive amp
    p9: drive frequency omega
    """
    # Unpack parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, omega = params

    # Scaling factors based on training ranges to keep basis terms O(1)
    # x range approx [-0.34, 0.36] -> scale 0.35
    # v range approx [-0.21, 0.17] -> scale 0.20
    x_scale = 0.35
    v_scale = 0.20
    
    xs = x / x_scale
    vs = v / v_scale
    
    # Precompute powers and products
    xs2 = xs * xs
    xs3 = xs2 * xs
    vs2 = vs * vs
    vs3 = vs2 * vs
    xs2_vs = xs2 * vs
    xs_vs = xs * vs
    
    # Drive terms
    st = np.sin(omega * t)
    ct = np.cos(omega * t)
    
    # Linear combination
    # Note: The coefficients p1, p2, etc. will adapt to the scaling.
    # The output is the acceleration a.
    return (
        p0
        + p1 * xs
        + p2 * xs3
        + p3 * vs
        + p4 * vs3
        + p5 * xs2_vs
        + p6 * xs_vs
        + p7 * st
        + p8 * ct
    )
