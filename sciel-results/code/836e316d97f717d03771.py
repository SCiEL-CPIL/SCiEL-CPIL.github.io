import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to roughly [-1, 1]
    # Domain is [0.5, 2.0], center 1.25, range 1.5, half-range 0.75
    xc = (x - 1.25) / 0.75
    yc = (y - 1.25) / 0.75
    zc = (z - 1.25) / 0.75

    # Build a richer basis that covers cubic and key cross terms.
    # We still use exactly 10 learnable parameters, but the basis is
    # non-linear in the inputs, so BFGS can fit a wider class of targets.
    b0 = np.ones_like(x)
    b1 = xc
    b2 = yc
    b3 = zc
    b4 = xc * xc * xc
    b5 = yc * yc * yc
    b6 = zc * zc * zc
    b7 = xc * yc * zc
    b8 = (xc * xc) * (yc * yc)
    b9 = (yc * yc) * (zc * zc)

    # Linear combination of the chosen basis functions
    return (
        params[0] * b0
        + params[1] * b1
        + params[2] * b2
        + params[3] * b3
        + params[4] * b4
        + params[5] * b5
        + params[6] * b6
        + params[7] * b7
        + params[8] * b8
        + params[9] * b9
    )
