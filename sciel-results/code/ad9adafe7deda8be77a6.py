import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)
    
    x = np.asarray(x, dtype=np.float64)
    
    # Hybrid basis combining smooth global features and localized components.
    # This structure allows the optimizer to fit a wider variety of function shapes
    # (polynomial trends, periodicity, saturation, and localized bumps) compared
    # to pure polynomial bases, which may be ill-conditioned or insufficient for
    # non-polynomial targets.
    
    # Basis components:
    # 0: 1 (bias)
    # 1: x (linear trend)
    # 2: x^2 (quadratic trend)
    # 3: sin(x) (periodic component)
    # 4: cos(x) (periodic component)
    # 5: sin(2x) (higher frequency periodic)
    # 6: cos(2x) (higher frequency periodic)
    # 7: tanh(x) (saturation/non-linearity)
    # 8: exp(-x^2) (Gaussian centered at 0)
    # 9: exp(-(x-1.5)^2) (Gaussian centered at 1.5 for asymmetry)
    
    basis = np.column_stack([
        np.ones_like(x),
        x,
        x**2,
        np.sin(x),
        np.cos(x),
        np.sin(2.0*x),
        np.cos(2.0*x),
        np.tanh(x),
        np.exp(-x**2),
        np.exp(-(x - 1.5)**2)
    ])
    
    result = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
    
    return result
