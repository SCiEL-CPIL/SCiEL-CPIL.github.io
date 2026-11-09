import numpy as np
import time

_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # x is expected to be in [0.1, 6]
    # params is an array of length 10
    
    # Strategy: Hybrid basis combining Fourier (for smooth/periodic components)
    # and rational/exponential terms (for non-periodic, singular-like behaviors).
    # This structure allows BFGS to fit a wider class of functions than pure
    # polynomials (conditioning issues) or pure Fourier (boundary discontinuities
    # for non-periodic functions).
    
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure safety: if x has bad values, return zeros
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(params)):
        return np.zeros_like(x)
    
    # Domain: [0.1, 6]
    # Normalize u to [0, 1] for Fourier terms
    L = 5.9
    u = (x - 0.1) / L
    
    # Basis functions:
    # 0: Constant
    # 1: Linear trend (helps with non-periodic drift)
    # 2: Fourier Cos(2*pi*u)
    # 3: Fourier Sin(2*pi*u)
    # 4: Fourier Cos(4*pi*u)
    # 5: Fourier Sin(4*pi*u)
    # 6: Fourier Cos(6*pi*u)
    # 7: Fourier Sin(6*pi*u)
    # 8: Rational 1/(1+x) (captures exponential decay-like or hyperbolic shapes)
    # 9: Log(1+x) (captures slow-growing non-polynomial trends)
    
    y = np.zeros_like(x)
    
    # 0: Constant
    y += params[0]
    
    # 1: Linear term
    y += params[1] * x
    
    # 2-3: Frequency 1
    y += params[2] * np.cos(2.0 * np.pi * u)
    y += params[3] * np.sin(2.0 * np.pi * u)
    
    # 4-5: Frequency 2
    y += params[4] * np.cos(4.0 * np.pi * u)
    y += params[5] * np.sin(4.0 * np.pi * u)
    
    # 6-7: Frequency 3
    y += params[6] * np.cos(6.0 * np.pi * u)
    y += params[7] * np.sin(6.0 * np.pi * u)
    
    # 8: Rational term
    # 1/(1+x) is smooth on [0.1, 6]
    y += params[8] / (1.0 + x)
    
    # 9: Log term
    # log(1+x) is smooth on [0.1, 6]
    y += params[9] * np.log1p(x)
    
    return y
