import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)

    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.flatten()
    p = np.asarray(params, dtype=np.float64)

    # Ensure p has exactly 10 elements
    if p.shape[0] != 10:
        p = p[:10] if p.shape[0] >= 10 else np.pad(p, (0, 10 - p.shape[0]))

    # Hybrid basis:
    # p[0]: 1 (constant)
    # p[1]: x
    # p[2]: x^2
    # p[3]: x^3
    # p[4]: x^4
    # p[5]: sin(x)
    # p[6]: cos(x)
    # p[7]: sin(2*x)
    # p[8]: cos(2*x)
    # p[9]: exp(-0.5 * x^2) (Gaussian bump for localized features)

    x2 = x * x
    result = (
        p[0] * 1.0
        + p[1] * x
        + p[2] * x2
        + p[3] * (x2 * x)
        + p[4] * (x2 * x2)
        + p[5] * np.sin(x)
        + p[6] * np.cos(x)
        + p[7] * np.sin(2.0 * x)
        + p[8] * np.cos(2.0 * x)
        + p[9] * np.exp(-0.5 * x2)
    )

    return result
