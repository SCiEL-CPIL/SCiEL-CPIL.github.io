# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for the acceleration of a damped nonlinear oscillator
    with time-dependent driving force.
    
    Strategy:
    1. Center and scale inputs to improve numerical conditioning for BFGS.
    2. Separate time-dependent driving terms from state-dependent dynamics.
    3. Include key nonlinear terms: cubic stiffness (Duffing), quadratic damping, 
       and cross-terms (x*v, x^2*v).
    
    Inputs:
        t: time array, range [30, 50]
        x: position array, range [-0.2056, 0.1749]
        v: velocity array, range [-0.3391, 0.3624]
        params: array of 10 fitted constants
    """
    # 1. Center and scale for conditioning
    # t: center 40, scale 10 -> range approx [-1, 1]
    tc = (t - 40.0) / 10.0
    # x: center -0.015, scale 0.2 -> range approx [-1, 1]
    xc = (x - (-0.015)) / 0.2
    # v: center 0.01, scale 0.35 -> range approx [-1, 1]
    vc = (v - 0.01) / 0.35

    # 2. Compute powers and cross terms
    tc2 = tc * tc
    tc3 = tc2 * tc
    
    xc2 = xc * xc
    xc3 = xc2 * xc
    
    vc2 = vc * vc
    
    xcv = xc * vc
    xc2v = xc2 * vc

    # 3. Linear combination of basis functions
    # p0: bias
    # p1-p3: time-dependent drive (polynomial)
    # p4: linear restoring force
    # p5: cubic nonlinearity (Duffing)
    # p6: linear damping
    # p7: quadratic damping
    # p8: cross term x*v
    # p9: cross term x^2*v (state-dependent damping correction)
    
    result = (
        params[0]
        + params[1] * tc
        + params[2] * tc2
        + params[3] * tc3
        + params[4] * xc
        + params[5] * xc3
        + params[6] * vc
        + params[7] * vc2
        + params[8] * xcv
        + params[9] * xc2v
    )

    return result
# EVOLVE-BLOCK-END
