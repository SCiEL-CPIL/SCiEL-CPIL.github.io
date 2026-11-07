import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation for symbolic regression with 2 inputs x, y.
    Uses a parametric trigonometric Fourier series with 10 basis functions.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # Basis functions: a mix of low-frequency trigonometric and polynomial terms
    # 1: bias
    b1 = np.ones_like(x)
    # 2: sin(x)
    b2 = np.sin(np.pi * x)
    # 3: cos(x)
    b3 = np.cos(np.pi * x)
    # 4: sin(y)
    b4 = np.sin(np.pi * y)
    # 5: cos(y)
    b5 = np.cos(np.pi * y)
    # 6: sin(x + y)
    b6 = np.sin(np.pi * (x + y))
    # 7: cos(x - y)
    b7 = np.cos(np.pi * (x - y))
    # 8: sin(2*pi*x)
    b8 = np.sin(2.0 * np.pi * x)
    # 9: sin(2*pi*y)
    b9 = np.sin(2.0 * np.pi * y)
    # 10: x*y (interaction term, captures non-separable structure)
    b10 = x * y

    pred = (p[0] * b1 +
            p[1] * b2 +
            p[2] * b3 +
            p[3] * b4 +
            p[4] * b5 +
            p[5] * b6 +
            p[7] * b7 +
            p[7] * b8 +
            p[8] * b9 +
            p[9] * b10)

    # Safety check for non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)

    return pred
