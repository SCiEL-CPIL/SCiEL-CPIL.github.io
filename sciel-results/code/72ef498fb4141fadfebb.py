"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _build_trig_design_matrix(x, y):
    """
    Build a design matrix using trigonometric basis functions.
    
    We use a mix of sine and cosine terms with different frequencies to capture
    periodic/oscillatory patterns that polynomials cannot represent efficiently.
    
    Basis terms (10 total):
    0: 1 (constant)
    1: cos(pi * x)
    2: sin(pi * x)
    3: cos(pi * y)
    4: sin(pi * y)
    5: cos(pi * x) * cos(pi * y)
    6: sin(pi * x) * sin(pi * y)
    7: cos(2*pi * x)
    8: sin(2*pi * y)
    9: cos(pi * x) * sin(pi * y)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Trigonometric basis functions
    F0 = np.ones_like(x)
    F1 = np.cos(np.pi * x)
    F2 = np.sin(np.pi * x)
    F3 = np.cos(np.pi * y)
    F4 = np.sin(np.pi * y)
    F5 = np.cos(np.pi * x) * np.cos(np.pi * y)
    F6 = np.sin(np.pi * x) * np.sin(np.pi * y)
    F7 = np.cos(2.0 * np.pi * x)
    F8 = np.sin(2.0 * np.pi * y)
    F9 = np.cos(np.pi * x) * np.sin(np.pi * y)
    
    return np.column_stack([F0, F1, F2, F3, F4, F5, F6, F7, F8, F9])

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict y using a linear combination of trigonometric basis features.
    The params are fitted by the evaluator using BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure params has length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    else:
        params = params[:10]
    
    # Build design matrix
    X = _build_trig_design_matrix(x, y)
    
    # Compute prediction: X @ params
    prediction = X @ params
    
    # Ensure prediction is finite
    if not np.all(np.isfinite(prediction)):
        return np.zeros_like(x)
    
    return prediction
# EVOLVE-BLOCK-END
