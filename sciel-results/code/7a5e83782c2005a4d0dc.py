"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    t_start = time.time()
    deadline = 145.0

    # Chebyshev polynomials of the first kind:
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1

    px0 = np.ones_like(x)
    px1 = x
    px2 = 2.0 * x**2 - 1.0
    px3 = 4.0 * x**3 - 3.0 * x

    py0 = np.ones_like(y)
    py1 = y
    py2 = 2.0 * y**2 - 1.0

    # 10 basis functions: separable product basis
    # x-degree 0-3, y-degree 0-2 gives 4*3 = 12 terms, we pick 10
    # b0 = T0(x)*T0(y) = 1
    # b1 = T1(x)*T0(y) = x
    # b2 = T0(x)*T1(y) = y
    # b3 = T2(x)*T0(y) = 2x^2-1
    # b4 = T0(x)*T2(y) = 2y^2-1
    # b5 = T1(x)*T1(y) = x*y
    # b6 = T3(x)*T0(y) = 4x^3-3x
    # b7 = T0(x)*T3(y) -- wait, we only have y up to degree 2
    # Let me restructure: 4 x-terms (deg 0-3) x 3 y-terms (deg 0-2) = 12
    # We need exactly 10. Drop the two highest-degree cross terms:
    # Drop T3(x)*T2(y) and T2(x)*T3(y) -- but we don't have T3(y)
    # Actually with x deg 0-3 and y deg 0-2, we have 12 terms.
    # Let's use: all 12 would be 12 params, but we only have 10.
    # Drop T3(x)*T1(y) and T3(x)*T2(y)? That seems arbitrary.
    # Better: use x deg 0-2 (3 terms) and y deg 0-3 (4 terms) = 12, drop 2.
    # Or: use a mix. Let me use 10 specific terms:
    #
    # Actually, let me just pick 10 terms that cover the space well:
    # T0*T0, T1*T0, T0*T1, T2*T0, T0*T2, T1*T1, T3*T0, T0*T3, T2*T1, T1*T2
    # That's 10 terms.
    
    py3 = 4.0 * y**3 - 3.0 * y

    b0 = px0 * py0       # 1
    b1 = px1 * py0       # x
    b2 = px0 * py1       # y
    b3 = px2 * py0       # 2x^2-1
    b4 = px0 * py2       # 2y^2-1
    b5 = px1 * py1       # x*y
    b6 = px3 * py0       # 4x^3-3x
    b7 = px0 * py3       # 4y^3-3y
    b8 = px2 * py1       # (2x^2-1)*y
    b9 = px1 * py2       # x*(2y^2-1)

    pred = (params[0] * b0 + params[1] * b1 + params[2] * b2 +
            params[3] * b3 + params[4] * b4 + params[5] * b5 +
            params[6] * b6 + params[7] * b7 + params[8] * b8 +
            params[9] * b9)

    if time.time() - t_start > deadline:
        return pred

    return pred
# EVOLVE-BLOCK-END
