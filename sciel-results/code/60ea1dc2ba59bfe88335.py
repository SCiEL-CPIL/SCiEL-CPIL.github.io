import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(x)

    p = params

    # Scaling to [-1, 1] based on known training ranges
    # This ensures good conditioning for the BFGS optimizer
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    x_range = x_max - x_min
    v_range = v_max - v_min
    
    if x_range < 1e-9:
        u = np.zeros_like(x)
    else:
        u = 2.0 * (x - x_min) / x_range - 1.0
        
    if v_range < 1e-9:
        w = np.zeros_like(v)
    else:
        w = 2.0 * (v - v_min) / v_range - 1.0

    # Precompute powers for efficiency and clarity
    u2 = u * u
    u3 = u2 * u
    w2 = w * w
    w3 = w2 * w

    # Basis functions:
    # We combine the physics-informed structure (oscillator terms) with a 
    # well-conditioned polynomial basis.
    #
    # The current program used: 1, u, w, u^2, u*w, w^2, u^3, u^2*w, w^3, u^4
    # Note that u^4 is redundant with u^2 in a linear least squares sense if 
    # the basis is complete enough, but here we have a fixed set.
    # However, u^4 is highly correlated with u^2 on [-1,1].
    # 
    # Inspiration 2 used: 1, u, w, u^2, w^2, u*w, u^3, w^3, u^2*w, u*w^2
    # This set is more balanced and includes u*w^2 which is important for 
    # nonlinear damping effects (e.g. -gamma * v * |v| or similar).
    #
    # Inspiration 3 used: 1, u, w, u^2, u*w, w^2, u^3, w^3, u^2*w, sin(u)
    # sin(u) is a strong candidate for restoring force if the oscillator is 
    # pendulum-like. However, sin(u) is very similar to u - u^3/6.
    # Given that we have u and u^3, sin(u) adds limited new information 
    # unless the nonlinearity is specifically sinusoidal.
    #
    # Let's adopt the basis from Inspiration 2, which is a complete 
    # polynomial basis up to total degree 3, except it swaps u^4 for u*w^2.
    # Total degree 3 terms are: u^3, u^2*w, u*w^2, w^3.
    # Degree 2 terms: u^2, u*w, w^2.
    # Degree 1 terms: u, w.
    # Degree 0 term: 1.
    # This gives exactly 10 terms:
    # 0: 1
    # 1: u
    # 2: w
    # 3: u^2
    # 4: u*w
    # 5: w^2
    # 6: u^3
    # 7: u^2*w
    # 8: u*w^2
    # 9: w^3
    
    b0 = np.ones_like(u)
    b1 = u
    b2 = w
    b3 = u2
    b4 = u * w
    b5 = w2
    b6 = u3
    b7 = u2 * w
    b8 = u * w2
    b9 = w3

    return (
        p[0] * b0 +
        p[1] * b1 +
        p[2] * b2 +
        p[3] * b3 +
        p[4] * b4 +
        p[5] * b5 +
        p[6] * b6 +
        p[7] * b7 +
        p[8] * b8 +
        p[9] * b9
    )
