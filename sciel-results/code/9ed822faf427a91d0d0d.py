"""Symbolic regression with 1 input using a hybrid Chebyshev-RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predict using a hybrid basis:
    - 4 Chebyshev polynomial terms (degrees 0-3) for global smooth trends.
    - 6 Gaussian (RBF) kernel terms for local non-polynomial features.
    
    params: length-10 array.
      - params[0:4] are Chebyshev coefficients c0, c1, c2, c3.
      - params[4:10] are RBF weights w1...w6.
    
    The RBF centers are fixed and evenly spaced across [0.1, 6.0].
    The RBF bandwidth is fixed at 0.6.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if len(x) == 0:
        return np.empty(0, dtype=np.float64)
    
    # --- 1. Global Trend: Chebyshev Polynomials ---
    # Map x from [0.1, 6.0] to t in [-1, 1]
    # t = (2*x - (max + min)) / (max - min)
    x_min, x_max = 0.1, 6.0
    t = (2.0 * x - (x_max + x_min)) / (x_max - x_min)
    t = np.clip(t, -1.0, 1.0)
    
    # Compute Chebyshev T_0 to T_3
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2*t^2 - 1
    # T_3(t) = 4*t^3 - 3*t
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    
    # Extract Chebyshev coefficients
    c0, c1, c2, c3 = params[0], params[1], params[2], params[3]
    
    global_part = c0 * T0 + c1 * T1 + c2 * T2 + c3 * T3
    
    # --- 2. Local Features: RBF Kernels ---
    # 6 centers evenly spaced in [0.1, 6.0]
    centers = np.linspace(0.1, 6.0, 6)
    
    # Bandwidth sigma. 
    # Distance between centers is (6.0-0.1)/5 = 1.18.
    # A sigma of 0.6 ensures significant overlap for smooth interpolation.
    sigma = 0.6
    
    # Compute RBF values: K(x, c_i) = exp(-0.5 * ((x - c_i) / sigma)^2)
    # Shape: (n_points, 6)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_vals = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Extract RBF weights
    weights = params[4:10]
    
    local_part = rbf_vals @ weights
    
    # --- 3. Combine ---
    prediction = global_part + local_part
    
    return prediction
# EVOLVE-BLOCK-END
