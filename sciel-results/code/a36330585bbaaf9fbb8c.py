import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    return (p[0] +
            p[1] * x +
            p[2] * y +
            p[3] * z +
            p[4] * x * y +
            p[5] * x * z +
            p[6] * y * z)
