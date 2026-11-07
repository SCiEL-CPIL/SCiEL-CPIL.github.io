import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Nonlinear oscillator acceleration using physics-informed bounded basis functions."""
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    x2 = x * x
    x3 = x2 * x
    sin_x = np.sin(x)
    tanh_x = np.tanh(x)
    v_abs = np.abs(v)
    v_quad = v * v_abs
    xv = x * v
    x2v = x2 * v
    sin_x_v = sin_x * v

    return (
        p0 * sin_x
        + p1 * x
        + p2 * x3
        + p3 * tanh_x
        + p4 * v
        + p5 * v_quad
        + p6 * xv
        + p7 * x2v
        + p8 * sin_x_v
        + p9
    )
