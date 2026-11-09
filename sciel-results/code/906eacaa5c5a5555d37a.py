"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -10, 10)
    
    # Nonlinear parametric model:
    # params[0:4] -> polynomial coefficients (degree 0-3)
    # params[4:6] -> sinusoidal: amplitude, frequency
    # params[6:9] -> Gaussian bump: center, width, amplitude
    # params[9]   -> global scale
    
    # Polynomial component (nonlinear in params through interaction with other terms)
    poly = (params[0] + params[1]*x + params[2]*x**2 + params[3]*x**3)
    
    # Sinusoidal component
    sin_term = params[4] * np.sin(params[5] * x)
    
    # Gaussian bump component (localized)
    center = params[6]
    width = np.abs(params[7]) + 0.1  # ensure positive width
    amp = params[8]
    bump = amp * np.exp(-((x - center) ** 2) / (2.0 * width ** 2))
    
    # Combine with global scale (nonlinear coupling)
    pred = params[9] * (poly + sin_term + bump)
    
    # Safety: replace any non-finite values
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
